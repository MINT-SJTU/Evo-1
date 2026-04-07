#!/usr/bin/env python

"""Patch legacy human-in-loop datasets so they can merge with newer rollout datasets."""

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from lerobot.configs import parser
from lerobot.datasets.dataset_tools import add_features
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.constants import ACTION
from lerobot.utils.utils import init_logging


MISSING_HIL_FEATURES = (
    "complementary_info.policy_action",
    "complementary_info.is_intervention",
    "complementary_info.state",
)


@dataclass
class PatchHilDatasetSchemaConfig:
    repo_id: str
    root: str | None = None
    output_repo_id: str | None = None
    output_dir: str | None = None


def _ensure_human_inloop_compatible_features(
    dataset_features: dict[str, dict],
    *,
    action_feature_names: list[str],
) -> None:
    dataset_features["complementary_info.policy_action"] = {
        "dtype": "float32",
        "shape": (len(action_feature_names),),
        "names": action_feature_names,
    }
    dataset_features["complementary_info.is_intervention"] = {
        "dtype": "float32",
        "shape": (1,),
        "names": ["is_intervention"],
    }
    dataset_features["complementary_info.state"] = {
        "dtype": "float32",
        "shape": (1,),
        "names": ["state"],
    }


def _build_missing_hil_features(dataset: LeRobotDataset) -> dict[str, tuple[np.ndarray, dict]]:
    action_feature = dataset.features[ACTION]
    action_names = action_feature["names"]
    if action_names is None:
        action_names = [f"action_{idx}" for idx in range(action_feature["shape"][0])]
    else:
        action_names = list(action_names)

    feature_defs: dict[str, dict] = {}
    _ensure_human_inloop_compatible_features(feature_defs, action_feature_names=action_names)

    num_action_dims = len(action_names)
    values_by_feature = {
        "complementary_info.policy_action": (
            lambda _row, _ep_idx, _frame_in_ep: np.zeros((num_action_dims,), dtype=np.float32)
        ),
        "complementary_info.is_intervention": (
            lambda _row, _ep_idx, _frame_in_ep: np.zeros((1,), dtype=np.float32)
        ),
        "complementary_info.state": (
            lambda _row, _ep_idx, _frame_in_ep: np.zeros((1,), dtype=np.float32)
        ),
    }

    missing_features = {}
    for feature_name in MISSING_HIL_FEATURES:
        if feature_name not in dataset.features:
            missing_features[feature_name] = (values_by_feature[feature_name], feature_defs[feature_name])

    return missing_features


@parser.wrap()
def patch_hil_dataset_schema(cfg: PatchHilDatasetSchemaConfig) -> LeRobotDataset:
    init_logging()

    dataset = LeRobotDataset(cfg.repo_id, root=cfg.root)
    missing_features = _build_missing_hil_features(dataset)
    if not missing_features:
        logging.info("Dataset already has a merge-compatible human-in-loop schema. Nothing to patch.")
        return dataset

    output_repo_id = cfg.output_repo_id or f"{cfg.repo_id}_hil_schema_patched"
    output_dir = Path(cfg.output_dir) if cfg.output_dir is not None else None
    if output_dir is None and cfg.root is not None:
        src_root = Path(cfg.root)
        output_dir = src_root.parent / f"{src_root.name}_hil_schema_patched"

    logging.info("Adding missing human-in-loop schema features: %s", sorted(missing_features))
    patched_dataset = add_features(
        dataset=dataset,
        features=missing_features,
        output_dir=output_dir,
        repo_id=output_repo_id,
    )
    logging.info("Patched dataset saved to %s", patched_dataset.root)
    return patched_dataset


def main():
    patch_hil_dataset_schema()


if __name__ == "__main__":
    main()
