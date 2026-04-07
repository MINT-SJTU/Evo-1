# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Human-in-loop recording helpers used by `lerobot_record.py`."""

from collections import deque
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.processor import PolicyAction, PolicyProcessorPipeline, RobotAction
from lerobot.rl.acp_tags import build_acp_tagged_task
from lerobot.robots import Robot
from lerobot.teleoperators import Teleoperator
from lerobot.utils.control_utils import predict_action


@dataclass
class ACPInferenceConfig:
    enable: bool = False
    use_cfg: bool = False
    cfg_beta: float = 1.0


POLICY_RUNTIME_STATE_KEYS = ("_action_queue", "_queues", "_prev_mean")


INTERVENTION_STATE_POLICY = 0.0
INTERVENTION_STATE_ACTIVE = 1.0
INTERVENTION_STATE_RELEASE = 2.0


def _get_torch_rng_state(device: torch.device) -> tuple[torch.Tensor, torch.Tensor | None]:
    cpu_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state(device) if device.type == "cuda" else None
    return cpu_state, cuda_state


def _set_torch_rng_state(
    device: torch.device, cpu_state: torch.Tensor, cuda_state: torch.Tensor | None
) -> None:
    torch.set_rng_state(cpu_state)
    if device.type == "cuda" and cuda_state is not None:
        torch.cuda.set_rng_state(cuda_state, device)


def _clone_runtime_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, deque):
        return deque((_clone_runtime_value(item) for item in value), maxlen=value.maxlen)
    if isinstance(value, dict):
        return {key: _clone_runtime_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_runtime_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_runtime_value(item) for item in value)
    return deepcopy(value)


def _capture_policy_runtime_state(policy: PreTrainedPolicy) -> dict[str, Any]:
    state: dict[str, Any] = {}
    for key in POLICY_RUNTIME_STATE_KEYS:
        if hasattr(policy, key):
            state[key] = _clone_runtime_value(getattr(policy, key))
    return state


def _restore_policy_runtime_state(policy: PreTrainedPolicy, state: dict[str, Any]) -> None:
    for key, value in state.items():
        setattr(policy, key, _clone_runtime_value(value))


def _predict_policy_action_with_runtime_state(
    *,
    observation_frame: dict[str, np.ndarray],
    policy: PreTrainedPolicy,
    device: torch.device,
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction],
    use_amp: bool,
    task: str | None,
    robot_type: str | None,
    runtime_state: dict[str, Any],
) -> PolicyAction:
    _restore_policy_runtime_state(policy, runtime_state)
    action = predict_action(
        observation=observation_frame,
        policy=policy,
        device=device,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        use_amp=use_amp,
        task=task,
        robot_type=robot_type,
    )
    runtime_state.clear()
    runtime_state.update(_capture_policy_runtime_state(policy))
    return action


def _predict_policy_action_with_acp_inference(
    *,
    observation_frame: dict[str, np.ndarray],
    policy: PreTrainedPolicy,
    device: torch.device,
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction],
    use_amp: bool,
    task: str | None,
    robot_type: str | None,
    acp_inference: ACPInferenceConfig,
    cond_runtime_state: dict[str, Any] | None = None,
    uncond_runtime_state: dict[str, Any] | None = None,
) -> PolicyAction:
    if not acp_inference.enable:
        return predict_action(
            observation=observation_frame,
            policy=policy,
            device=device,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            use_amp=use_amp,
            task=task,
            robot_type=robot_type,
        )

    conditional_task = build_acp_tagged_task(task, is_positive=True)
    if not acp_inference.use_cfg:
        return predict_action(
            observation=observation_frame,
            policy=policy,
            device=device,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            use_amp=use_amp,
            task=conditional_task,
            robot_type=robot_type,
        )

    if cond_runtime_state is None or uncond_runtime_state is None:
        raise ValueError("CFG inference requires cond/uncond runtime states.")

    cpu_state, cuda_state = _get_torch_rng_state(device)
    action_cond = _predict_policy_action_with_runtime_state(
        observation_frame=observation_frame,
        policy=policy,
        device=device,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        use_amp=use_amp,
        task=conditional_task,
        robot_type=robot_type,
        runtime_state=cond_runtime_state,
    )
    _set_torch_rng_state(device, cpu_state, cuda_state)
    action_uncond = _predict_policy_action_with_runtime_state(
        observation_frame=observation_frame,
        policy=policy,
        device=device,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        use_amp=use_amp,
        task=task,
        robot_type=robot_type,
        runtime_state=uncond_runtime_state,
    )
    return action_uncond + acp_inference.cfg_beta * (action_cond - action_uncond)


class PolicySyncDualArmExecutor:
    """Broadcast one policy-derived robot action to follower + teleop arm."""

    def __init__(self, robot: Robot, teleop: Teleoperator, parallel_dispatch: bool = True):
        self.robot = robot
        self.teleop = teleop
        self.parallel_dispatch = parallel_dispatch
        self._pool = ThreadPoolExecutor(max_workers=2) if parallel_dispatch else None

    def send_action(self, action: RobotAction) -> RobotAction:
        if self._pool is None:
            sent_action = self.robot.send_action(action)
            self.teleop.send_feedback(action)
            return sent_action

        robot_future = self._pool.submit(self.robot.send_action, action)
        teleop_future = self._pool.submit(self.teleop.send_feedback, action)
        sent_action = robot_future.result()
        teleop_future.result()
        return sent_action

    def shutdown(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=True)
