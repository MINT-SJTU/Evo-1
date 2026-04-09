"""Utility to peek Lerobot v2.1 datasets and inspect normalization stats."""

from __future__ import annotations

import argparse
import json
import logging
import yaml
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Tuple
from tqdm import tqdm

import numpy as np
import pandas as pd

from .dataset_process_suite import get_suite, ProcessedData

REQUIRED_METRICS: Tuple[str, ...] = ("std", "mean", "min", "max", "q01", "q99")


def compute_normstats(
    dataset_path: Path, 
    use_delta_actions: bool = True, 
    action_horizon: int = 50, 
    dataset_config: Dict = None
) -> Dict[str, object]:
    
    meta_dir = dataset_path / "meta"
    data_dir = dataset_path / "data"
    info_path = meta_dir / "info.json"

    if not info_path.exists():
        raise FileNotFoundError(f"Missing meta/info.json under {dataset_path}")

    if dataset_config is None:
        dataset_config = {}

    info = json.loads(info_path.read_text())
    print(f"Inspecting dataset at {dataset_path}")
    print(
        "Meta summary:"
        f" {info.get('codebase_version', 'unknown')} |"
        f" episodes={info.get('total_episodes', 'n/a')} |"
        f" frames={info.get('total_frames', 'n/a')}"
    )

    parquet_files: List[Path] = []
    for chunk in sorted(data_dir.glob("chunk-*/")):
        parquet_files.extend(sorted(chunk.glob("episode_*.parquet")))

    if not parquet_files:
        raise FileNotFoundError(f"No episode parquet files located under {data_dir}")

    total_frames = 0
    per_episode: Dict[int, int] = {}
    action_dim = None
    state_dim = None
    
    action_batches: List[np.ndarray] = []
    state_batches: List[np.ndarray] = []

    suite_name = dataset_config.get('process_suite', 'default')
    suite_config = dataset_config.get('suite_config', {})
    print(f"Using Process Suite: '{suite_name}' with config: {suite_config}")
    
    try:
        suite = get_suite(suite_name, suite_config)
    except Exception as e:
        raise RuntimeError(f"Failed to initialize suite '{suite_name}': {e}")
    
    print(f"Processing {len(parquet_files)} episodes...")
    if use_delta_actions:
        print(f"Computing relative action statistics (via Suite) with horizon {action_horizon}...")
    else:
        print("Computing absolute action statistics (via Suite)...")

    for pq_path in tqdm(parquet_files, desc=f"Processing {dataset_path.name}", unit="ep"):
        df = pd.read_parquet(pq_path)
        frames_here = len(df)
        total_frames += frames_here

        try:
            episode_idx = int(pq_path.stem.split("_")[-1])
        except ValueError:
            episode_idx = len(per_episode)
        per_episode[episode_idx] = frames_here

        if frames_here > 0:
            # Keep padding behavior aligned with dataset sampling.
            last_row = df.iloc[-1:]
            padding_rows = pd.concat([last_row] * action_horizon, ignore_index=True)
            df_padded = pd.concat([df, padding_rows], ignore_index=True)
            
            episode_actions = []
            episode_states = []

            for i in range(len(df_padded) - action_horizon + 1):
                sub_df = df_padded.iloc[i : i + action_horizon]

                processed_data: ProcessedData = suite.process(sub_df, use_delta_action=use_delta_actions)

                if processed_data.state is not None:
                    episode_states.append(processed_data.state)

                episode_actions.append(processed_data.actions)
                
                if action_dim is None:
                    action_dim = processed_data.action_dim
                if state_dim is None:
                    state_dim = processed_data.state_dim

            if episode_actions:
                ep_act_np = np.stack(episode_actions)
                ep_act_flat = ep_act_np.reshape(-1, action_dim)
                action_batches.append(ep_act_flat)
            
            if episode_states:
                ep_state_np = np.stack(episode_states)
                state_batches.append(ep_state_np)

    lengths = list(per_episode.values())
    length_stats = {
        "min": min(lengths) if lengths else 0,
        "max": max(lengths) if lengths else 0,
        "avg": round(mean(lengths), 2) if lengths else 0,
    }

    print(
        f"Loaded {len(per_episode)} episodes"
        f" with {total_frames} frames."
    )
    print(
        f"Action rows (flattened windows): {sum(len(b) for b in action_batches)} | dim={action_dim}"
        f"\nState rows:  {sum(len(b) for b in state_batches)} | dim={state_dim}"
    )

    actions = _concat_batches(action_batches)
    states = _concat_batches(state_batches)
    
    stats_payload = {
        "action": _vector_stats(actions),
        "observation.state": _vector_stats(states),
    }

    stats_path = meta_dir / "stats.json"
    stats_path.write_text(
        json.dumps(stats_payload, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote global stats (includes q01/q99) to {stats_path}.")

    per_episode_stats = meta_dir / "episodes_stats.jsonl"
    present_metrics: Iterable[str] = tuple()
    missing_metrics: Iterable[str] = REQUIRED_METRICS
    if per_episode_stats.exists():
        present_metrics, missing_metrics = _collect_metric_keys(per_episode_stats)
    else:
        print("episodes_stats.jsonl not found; only global stats were produced.")

    return {
        "dataset_root": str(dataset_path),
        "episodes_found": len(per_episode),
        "frames_read": total_frames,
        "action_dim": action_dim,
        "state_dim": state_dim,
        "episode_length_stats": length_stats,
        "stats_file": stats_path.name,
    }


def _concat_batches(batches: List[np.ndarray]) -> np.ndarray:
    if not batches:
        return np.empty((0, 0), dtype=np.float32)
    return np.concatenate(batches, axis=0)


def _vector_stats(array: np.ndarray) -> Dict[str, List[float]]:
    if array.size == 0:
        return {metric: [] for metric in REQUIRED_METRICS}

    arr = array.astype(np.float64, copy=False)
    stats = {
        "mean": arr.mean(axis=0).tolist(),
        "std": arr.std(axis=0, ddof=0).tolist(),
        "min": arr.min(axis=0).tolist(),
        "max": arr.max(axis=0).tolist(),
        "q01": np.quantile(arr, 0.01, axis=0).tolist(),
        "q99": np.quantile(arr, 0.99, axis=0).tolist(),
    }
    return stats


def _collect_metric_keys(stats_path: Path) -> Tuple[List[str], List[str]]:
    present = set()
    with stats_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            payload = json.loads(line)
            stats_section = payload.get("stats")
            if not isinstance(stats_section, dict):
                continue
            if stats_section and all(isinstance(v, dict) for v in stats_section.values()):
                stats_dicts = stats_section.values()
            else:
                stats_dicts = [stats_section]
            for entry in stats_dicts:
                if not isinstance(entry, dict):
                    continue
                for key in entry.keys():
                    if key in REQUIRED_METRICS:
                        present.add(key)
            if len(present) == len(REQUIRED_METRICS):
                break
    missing = [m for m in REQUIRED_METRICS if m not in present]
    return sorted(present), missing


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect lerobot dataset stats.")
    parser.add_argument(
        "config_path",
        type=str,
        help="The path to the dataset configuration file (e.g., config.yaml).",
    )
    parser.add_argument(
        "--action_horizon",
        type=int,
        default=50,
        help="The length of action sequences (action_horizon)."
    )
    args = parser.parse_args()
    config_path = Path(args.config_path)
    if not config_path.exists():
        logging.error(f"Configuration file does not exist: {config_path}")
        return

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    for arm_name, arm_config in config.get('data_groups', {}).items():
        for dataset_name, dataset_config in arm_config.items():
            path_str = dataset_config.get('path')
            if not path_str:
                logging.warning(f"Dataset '{arm_name}/{dataset_name}' has no path configured, skipping.")
                continue
            
            dataset_path = Path(path_str)
            use_delta = dataset_config.get('use_delta_action', False)

            compute_normstats(
                dataset_path, 
                use_delta_actions=use_delta, 
                action_horizon=args.action_horizon, 
                dataset_config=dataset_config
            )


if __name__ == "__main__":
    main()