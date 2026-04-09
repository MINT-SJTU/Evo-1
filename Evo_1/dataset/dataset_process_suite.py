# Extensible dataset processing suites for various robotics datasets.

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional, List
from dataclasses import dataclass
import logging


@dataclass
class ProcessedData:
    """Container returned by a suite after processing one horizon window."""
    state: Optional[np.ndarray]  # Initial state at t=0, used for normalization stats.
    actions: np.ndarray          # Action sequence with shape [horizon, action_dim].
    state_dim: int               # Dimension of `state` (0 if state is unavailable).
    action_dim: int              # Number of action channels.


class BaseProcessSuite(ABC):
    """Base class for all dataset-specific processing suites.

    A suite defines how to:
    1) read state/action fields from raw dataset rows,
    2) optionally convert absolute actions into relative (delta) actions,
    3) return a consistent `ProcessedData` object used by downstream stats code.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Args:
            config: Suite-specific configuration dictionary.
        """
        self.config = config or {}
    
    @abstractmethod
    def extract_state(self, row: pd.Series) -> Optional[np.ndarray]:
        """Extract state from one row (typically the first row in a horizon window)."""
        pass
    
    @abstractmethod
    def extract_actions(self, sub_df: pd.DataFrame) -> np.ndarray:
        """Extract action sequence from a horizon-sized dataframe window."""
        pass
    
    def compute_relative_actions(
        self, 
        actions: np.ndarray, 
        states: np.ndarray,
        relative_dims: List[Tuple[int, int]]
    ) -> np.ndarray:
        """
        Convert selected action dimensions from absolute values to relative values.
        
        Args:
            actions: Raw action array with shape [horizon, action_dim].
            states: State array with shape [horizon, state_dim].
            relative_dims: Inclusive-exclusive index ranges to convert, e.g.
                [(0, 6), (7, 13)].
        
        Returns:
            Relative action array with shape [horizon, action_dim].
        """
        relative_actions = actions.copy()
        
        for start, end in relative_dims:
            if actions.shape[1] >= end and states.shape[1] >= end:
                relative_actions[:, start:end] = actions[:, start:end] - states[:, start:end]
        
        return relative_actions
    
    def process(
        self, 
        sub_df: pd.DataFrame, 
        use_delta_action: bool = True
    ) -> ProcessedData:
        """
        Main processing entry for one horizon window.
        
        Args:
            sub_df: Dataframe slice containing `action_horizon` rows.
            use_delta_action: Whether to apply suite-defined relative conversion.
        
        Returns:
            A `ProcessedData` instance with state, actions, and dimensions.
        """
        # Read the reference state from the first timestep.
        init_state = self.extract_state(sub_df.iloc[0])
        
        # Read the action sequence for this horizon window.
        actions = self.extract_actions(sub_df)
        
        # Optionally transform absolute actions into relative actions.
        if use_delta_action and init_state is not None:
            actions = self._convert_to_relative(sub_df, actions)
        
        return ProcessedData(
            state=init_state,
            actions=actions.tolist() if isinstance(actions, np.ndarray) else actions,
            state_dim=len(init_state) if init_state is not None else 0,
            action_dim=actions.shape[1] if isinstance(actions, np.ndarray) else len(actions[0])
        )
    
    def _convert_to_relative(
        self, 
        sub_df: pd.DataFrame, 
        actions: np.ndarray
    ) -> np.ndarray:
        """Convert actions to relative form (subclasses can override custom logic)."""
        return actions



class DefaultSuite(BaseProcessSuite):
    """
    Default suite that reads `observation.state` and `action` directly.

    Relative conversion policy:
    - If at least 6 dimensions exist, convert [0:6] (typically end-effector pose).
    - If at least 13 dimensions exist, also convert [7:13] (e.g., second arm).
    - Gripper dimensions are intentionally not modified by these slices.
    """
    
    def extract_state(self, row: pd.Series) -> Optional[np.ndarray]:
        state = row.get("observation.state", None)
        if state is not None:
            return np.array(state)
        return None
    
    def extract_actions(self, sub_df: pd.DataFrame) -> np.ndarray:
        return np.stack(sub_df["action"].to_list())
    
    def _convert_to_relative(
        self, 
        sub_df: pd.DataFrame, 
        actions: np.ndarray
    ) -> np.ndarray:
        states = np.stack(sub_df["observation.state"].to_list())
        
        # Default relative ranges: primary EE pose and optional secondary arm.
        relative_dims = []
        action_dim = actions.shape[1]
        state_dim = states.shape[1]
        
        if action_dim >= 6 and state_dim >= 6:
            relative_dims.append((0, 6))
        if action_dim >= 13 and state_dim >= 13:
            relative_dims.append((7, 13))
        
        return self.compute_relative_actions(actions, states, relative_dims)


class FrankaEEPoseSuite(BaseProcessSuite):
    """
        Example suite for Franka datasets represented in end-effector pose space.

        Typical use cases:
        - LIBERO
        - austin_buds
        - austin_sirius

        Expected schema (default):
        - State key: `observation.state`
            usually 7D = [x, y, z, r, p, y, gripper]
        - Action key: `action`
            usually 7D = [x, y, z, r, p, y, gripper] or corresponding command values

        Relative conversion behavior:
        - Only EE pose dimensions in `ee_pose_dims` (default [0:6]) are converted
            via `action - state`.
        - Gripper channel(s) remain absolute on purpose. This avoids changing the
            semantics of open/close commands.

        Configuration keys:
        - state_key: column name for state vector
        - action_key: column name for action vector
        - ee_pose_dims: tuple(start, end) defining EE pose dimensions to convert
        - gripper_indices: optional documentation field for gripper positions
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.state_key = self.config.get("state_key", "observation.state")
        self.action_key = self.config.get("action_key", "action")
        self.gripper_indices = self.config.get("gripper_indices", [6])  # Gripper index/indices.
        self.ee_pose_dims = self.config.get("ee_pose_dims", (0, 6))  # EE pose dimensions.
    
    def extract_state(self, row: pd.Series) -> Optional[np.ndarray]:
        state = row.get(self.state_key, None)
        if state is not None:
            return np.array(state)
        return None
    
    def extract_actions(self, sub_df: pd.DataFrame) -> np.ndarray:
        return np.stack(sub_df[self.action_key].to_list())
    
    def _convert_to_relative(
        self, 
        sub_df: pd.DataFrame, 
        actions: np.ndarray
    ) -> np.ndarray:
        try:
            states = np.stack(sub_df[self.state_key].to_list())
        except Exception:
            logging.warning(f"Cannot extract states from {self.state_key}, skipping relative conversion")
            return actions
        
        relative_actions = actions.copy()
        start, end = self.ee_pose_dims
        
        if actions.shape[1] >= end and states.shape[1] >= end:
            relative_actions[:, start:end] = actions[:, start:end] - states[:, start:end]
        
        # Gripper values are intentionally left unchanged.
        
        return relative_actions


class FrankaJointAngleSuite(BaseProcessSuite):
    """
        Example suite for Franka datasets represented in joint-angle space.

        Typical use cases:
        - RLBench
        - berkeley_rpt

        Expected schema (default):
        - State key: `observation.state`
            often 8D = [joint_0..joint_6, gripper]
        - Action key: `action`
            often 8D = [joint_0..joint_6, gripper]

        Relative conversion behavior:
        - Convert joint dimensions [0:joint_dims] using `action - state`.
        - Keep any remaining channels (commonly gripper) unchanged.

        Configuration keys:
        - state_key: column name for state vector
        - action_key: column name for action vector
        - joint_dims: number of leading joint dimensions to convert
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.state_key = self.config.get("state_key", "observation.state")
        self.action_key = self.config.get("action_key", "action")
        self.joint_dims = self.config.get("joint_dims", 7)
    
    def extract_state(self, row: pd.Series) -> Optional[np.ndarray]:
        state = row.get(self.state_key, None)
        if state is not None:
            return np.array(state)
        return None
    
    def extract_actions(self, sub_df: pd.DataFrame) -> np.ndarray:
        return np.stack(sub_df[self.action_key].to_list())
    
    def _convert_to_relative(
        self, 
        sub_df: pd.DataFrame, 
        actions: np.ndarray
    ) -> np.ndarray:
        try:
            states = np.stack(sub_df[self.state_key].to_list())
        except Exception:
            return actions
        
        # Convert leading joint dimensions; keep trailing channels unchanged.
        relative_dims = [(0, min(self.joint_dims, actions.shape[1], states.shape[1]))]
        return self.compute_relative_actions(actions, states, relative_dims)


class DroidEEFSuite(BaseProcessSuite):
    """
        Example suite for DROID datasets with split Cartesian and gripper fields.

        Typical use case:
        - droid_101_eef

        Expected schema (default):
        - State position key: `observation.state.cartesian_position` (6D)
        - State gripper key: `observation.state.gripper_position` (scalar)
        - Action position key: `action.cartesian_position` (6D)
        - Action gripper key: `action.gripper_position` (scalar)

        Key preprocessing behavior:
        - Build a unified 7D state/action by concatenating Cartesian(6D) + gripper(1D).
        - Invert gripper with `1 - value` to match the convention used by this project.
        - During relative conversion, only Cartesian dimensions [0:6] are converted.
            Gripper stays absolute after inversion.

        Configuration keys:
        - state_key, action_key
        - state_gripper_key, action_gripper_key
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.state_key = self.config.get("state_key", "observation.state.cartesian_position")
        self.action_key = self.config.get("action_key", "action.cartesian_position")
        self.action_gripper_key = self.config.get("action_gripper_key", "action.gripper_position")
        self.state_gripper_key = self.config.get("state_gripper_key", "observation.state.gripper_position")
    
    def extract_state(self, row: pd.Series) -> Optional[np.ndarray]:
        state_pos = row.get(self.state_key, None)
        state_gripper_raw = row.get(self.state_gripper_key, None)
        if state_pos is not None and state_gripper_raw is not None:
            state_gripper = 1 - state_gripper_raw
            return np.concatenate([np.array(state_pos), np.array([state_gripper])], axis=0)
        return None
    
    def extract_actions(self, sub_df: pd.DataFrame) -> np.ndarray:
        cart_actions = np.stack(sub_df[self.action_key].to_list())
        
        gripper_raw = sub_df[self.action_gripper_key].to_list()
        gripper = 1 - np.array(gripper_raw).reshape(-1, 1)

        return np.concatenate([cart_actions, gripper], axis=1)
    
    def _convert_to_relative(
        self, 
        sub_df: pd.DataFrame, 
        actions: np.ndarray
    ) -> np.ndarray:
        try:
            states = np.stack(sub_df[self.state_key].to_list())
        except Exception:
            return actions
        
        relative_actions = actions.copy()
        # Convert only the first 6 Cartesian dimensions to relative values.
        if actions.shape[1] >= 6 and states.shape[1] >= 6:
            relative_actions[:, :6] = actions[:, :6] - states[:, :6]
        
        return relative_actions


class AlohaJointAngleSuite(BaseProcessSuite):
    """
    ALOHA dual-arm joint-angle suite.

    Typical use cases:
    - RoboTwin
    - ALOHA datasets

    Typical shape:
    - 14D = 7D left arm + 7D right arm
    - Index 6 and 13 are commonly grippers and remain absolute
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.state_key = self.config.get("state_key", "observation.state")
        self.action_key = self.config.get("action_key", "action")
        self.arm_dims = self.config.get("arm_dims", 7)  # Number of dimensions per arm.
        self.gripper_indices = self.config.get("gripper_indices", [6, 13])
    
    def extract_state(self, row: pd.Series) -> Optional[np.ndarray]:
        state = row.get(self.state_key, None)
        if state is not None:
            return np.array(state)
        return None
    
    def extract_actions(self, sub_df: pd.DataFrame) -> np.ndarray:
        return np.stack(sub_df[self.action_key].to_list())
    
    def _convert_to_relative(
        self, 
        sub_df: pd.DataFrame, 
        actions: np.ndarray
    ) -> np.ndarray:
        try:
            states = np.stack(sub_df[self.state_key].to_list())
        except Exception:
            return actions
        
        relative_actions = actions.copy()
        action_dim = actions.shape[1]
        state_dim = states.shape[1]
        
        # Left arm joints (0-5), skip gripper at 6.
        if action_dim >= 6 and state_dim >= 6:
            relative_actions[:, :6] = actions[:, :6] - states[:, :6]
        
        # Right arm joints (7-12), skip gripper at 13.
        if action_dim >= 13 and state_dim >= 13:
            relative_actions[:, 7:13] = actions[:, 7:13] - states[:, 7:13]
        
        return relative_actions


class MetaWorldSuite(BaseProcessSuite):
    """
    MetaWorld dataset suite.

    Characteristics:
    - State: observation.state
    - Action: action (typically 4D: delta xyz + gripper)
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.state_key = self.config.get("state_key", "observation.state")
        self.action_key = self.config.get("action_key", "action")
    
    def extract_state(self, row: pd.Series) -> Optional[np.ndarray]:
        state = row.get(self.state_key, None)
        if state is not None:
            return np.array(state)
        return None
    
    def extract_actions(self, sub_df: pd.DataFrame) -> np.ndarray:
        return np.stack(sub_df[self.action_key].to_list())
    
    def _convert_to_relative(
        self, 
        sub_df: pd.DataFrame, 
        actions: np.ndarray
    ) -> np.ndarray:
        # MetaWorld actions are usually already delta commands.
        return actions


class CustomSuite(BaseProcessSuite):
    """
    Fully configurable suite driven by config fields.

    Configuration example:
    {
        "state_key": "observation.state",
        "action_key": "action",
        "action_concat_keys": ["action.cartesian", "action.gripper"],  # Optional: concatenate multiple action fields.
        "relative_dims": [[0, 6], [7, 13]],  # Optional: dimensions to convert to relative values.
        "gripper_indices": [6, 13]  # Optional: gripper indices (kept absolute by suite logic).
    }
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.state_key = self.config.get("state_key", "observation.state")
        self.action_key = self.config.get("action_key", "action")
        self.action_concat_keys = self.config.get("action_concat_keys", None)
        self.relative_dims = self.config.get("relative_dims", [[0, 6]])
        self.gripper_indices = self.config.get("gripper_indices", [])
    
    def extract_state(self, row: pd.Series) -> Optional[np.ndarray]:
        state = row.get(self.state_key, None)
        if state is not None:
            return np.array(state)
        return None
    
    def extract_actions(self, sub_df: pd.DataFrame) -> np.ndarray:
        if self.action_concat_keys:
            arrays = []
            for key in self.action_concat_keys:
                arr = np.stack(sub_df[key].to_list())
                if arr.ndim == 1:
                    arr = arr.reshape(-1, 1)
                arrays.append(arr)
            return np.concatenate(arrays, axis=1)
        else:
            return np.stack(sub_df[self.action_key].to_list())
    
    def _convert_to_relative(
        self, 
        sub_df: pd.DataFrame, 
        actions: np.ndarray
    ) -> np.ndarray:
        try:
            states = np.stack(sub_df[self.state_key].to_list())
        except Exception:
            return actions
        
        relative_dims = [tuple(d) for d in self.relative_dims]
        return self.compute_relative_actions(actions, states, relative_dims)



# suite registry
SUITE_REGISTRY: Dict[str, type] = {
    "default": DefaultSuite,
    "franka_ee_pose": FrankaEEPoseSuite,
    "franka_joint_angle": FrankaJointAngleSuite,
    "droid_eef": DroidEEFSuite,
    "aloha_joint_angle": AlohaJointAngleSuite,
    "metaworld": MetaWorldSuite,
    "custom": CustomSuite,
}


def register_suite(name: str, suite_class: type):
    """Register a new suite"""
    if not issubclass(suite_class, BaseProcessSuite):
        raise TypeError(f"Suite class must inherit from BaseProcessSuite")
    SUITE_REGISTRY[name] = suite_class


def get_suite(name: str, config: Dict[str, Any] = None) -> BaseProcessSuite:
    """
    Create a suite instance by name.
    
    Args:
        name: Suite registry name.
        config: Suite configuration dictionary.
    
    Returns:
        Suite instance.
    """
    if name not in SUITE_REGISTRY:
        available = list(SUITE_REGISTRY.keys())
        raise ValueError(f"Unknown suite '{name}'. Available suites: {available}")
    
    return SUITE_REGISTRY[name](config)


def list_available_suites() -> List[str]:
    """List all registered suite names."""
    return list(SUITE_REGISTRY.keys())