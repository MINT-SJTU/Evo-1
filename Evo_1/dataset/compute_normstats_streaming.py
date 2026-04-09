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


class RunningStats:
    """Compute streaming stats using Welford + reservoir sampling."""

    def __init__(self, dim: int, reservoir_size: int = 50_000_000):
        self.n = 0
        self.dim = dim
        self.mean = np.zeros(dim, dtype=np.float64)
        self.M2 = np.zeros(dim, dtype=np.float64)
        self.min_val = np.full(dim, np.inf, dtype=np.float64)
        self.max_val = np.full(dim, -np.inf, dtype=np.float64)

        self.reservoir_size = reservoir_size
        self.reservoir = np.zeros((reservoir_size, dim), dtype=np.float32)
        self.reservoir_idx = 0

    def update(self, batch: np.ndarray):
        """
        batch: (B, D) array
        """
        if batch.size == 0:
            return

        batch = batch.astype(np.float64)
        B = batch.shape[0]

        self.min_val = np.minimum(self.min_val, np.min(batch, axis=0))
        self.max_val = np.maximum(self.max_val, np.max(batch, axis=0))

        batch_mean = np.mean(batch, axis=0)
        batch_m2 = np.sum((batch - batch_mean) ** 2, axis=0)
        batch_n = B

        delta = batch_mean - self.mean
        new_n = self.n + batch_n

        self.M2 += batch_m2 + delta**2 * self.n * batch_n / new_n
        self.mean += delta * batch_n / new_n
        self.n = new_n

        # Fill reservoir first, then run replacement sampling.
        if self.reservoir_idx < self.reservoir_size:
            space = self.reservoir_size - self.reservoir_idx
            take = min(space, B)
            self.reservoir[self.reservoir_idx : self.reservoir_idx + take] = batch[
                :take
            ].astype(np.float32)
            self.reservoir_idx += take

            remaining_batch = batch[take:]
            remaining_start_idx = self.n - B + take
        else:
            remaining_batch = batch
            remaining_start_idx = self.n - B

        if len(remaining_batch) > 0:
            current_total = remaining_start_idx
            for i in range(len(remaining_batch)):
                current_total += 1
                r = np.random.randint(0, current_total)
                if r < self.reservoir_size:
                    self.reservoir[r] = remaining_batch[i].astype(np.float32)

    def compute(self) -> Dict[str, List[float]]:
        if self.n == 0:
            return {k: [] for k in REQUIRED_METRICS}

        std = np.sqrt(self.M2 / self.n)

        valid_reservoir = self.reservoir[: self.reservoir_idx]
        q01 = np.quantile(valid_reservoir, 0.01, axis=0)
        q99 = np.quantile(valid_reservoir, 0.99, axis=0)

        return {
            "mean": self.mean.tolist(),
            "std": std.tolist(),
            "min": self.min_val.tolist(),
            "max": self.max_val.tolist(),
            "q01": q01.tolist(),
            "q99": q99.tolist(),
        }


def compute_normstats(
    dataset_path: Path,
    use_delta_actions: bool = True,
    action_horizon: int = 50,
    dataset_config: Dict = None,
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

    # Initialize Streaming Stats
    action_stats = None
    state_stats = None

    suite_name = dataset_config.get("process_suite", "default")
    suite_config = dataset_config.get("suite_config", {})
    print(f"Using Process Suite: '{suite_name}' with config: {suite_config}")

    try:
        suite = get_suite(suite_name, suite_config)
    except Exception as e:
        raise RuntimeError(f"Failed to initialize suite '{suite_name}': {e}")

    print(f"Processing {len(parquet_files)} episodes using Streaming method...")
    if use_delta_actions:
        print(
            f"Computing relative action statistics (via Suite) with horizon {action_horizon}..."
        )
    else:
        print("Computing absolute action statistics (via Suite)...")

    for pq_path in tqdm(
        parquet_files, desc=f"Processing {dataset_path.name}", unit="ep"
    ):
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

                processed_data: ProcessedData = suite.process(
                    sub_df, use_delta_action=use_delta_actions
                )

                episode_actions.append(processed_data.actions)

                if processed_data.state is not None:
                    episode_states.append(processed_data.state)

            if episode_actions:
                actions_np = np.stack(episode_actions)
                action_dim = actions_np.shape[2]

                actions_flat = actions_np.reshape(-1, action_dim)

                if action_stats is None:
                    action_stats = RunningStats(dim=action_dim)

                action_stats.update(actions_flat)

            if episode_states:
                states_np = np.stack(episode_states)
                state_dim = states_np.shape[1]

                if state_stats is None:
                    state_stats = RunningStats(dim=state_dim)

                state_stats.update(states_np)

    lengths = list(per_episode.values())
    length_stats = {
        "min": min(lengths) if lengths else 0,
        "max": max(lengths) if lengths else 0,
        "avg": round(mean(lengths), 2) if lengths else 0,
    }

    print(f"Loaded {len(per_episode)} episodes" f" with {total_frames} frames.")
    print(
        "Episode length stats (frames):"
        f" min={length_stats['min']}"
        f" max={length_stats['max']}"
        f" mean={length_stats['avg']}"
    )

    stats_payload = {
        "action": action_stats.compute() if action_stats else _empty_stats(),
        "observation.state": state_stats.compute() if state_stats else _empty_stats(),
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
        print(
            f"Per-episode stats file: {per_episode_stats.name}."
            " Metrics present: " + ", ".join(present_metrics)
        )
        if missing_metrics:
            print(
                "Missing targets (per-episode file): "
                + ", ".join(
                    metric for metric in REQUIRED_METRICS if metric in missing_metrics
                )
            )
        else:
            print("Per-episode stats already cover all required metrics.")
    else:
        print("episodes_stats.jsonl not found; only global stats were produced.")

    return {
        "dataset_root": str(dataset_path),
        "episodes_found": len(per_episode),
        "frames_read": total_frames,
        "action_dim": action_stats.dim if action_stats else 0,
        "state_dim": state_stats.dim if state_stats else 0,
        "episode_length_stats": length_stats,
        "stats_file": stats_path.name,
        "metrics_present": list(present_metrics),
        "metrics_missing": list(missing_metrics),
    }


def _empty_stats() -> Dict[str, List[float]]:
    return {metric: [] for metric in REQUIRED_METRICS}


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

            if stats_section and all(
                isinstance(v, dict) for v in stats_section.values()
            ):
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

    missing = [metric for metric in REQUIRED_METRICS if metric not in present]
    return sorted(present), missing


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


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
        help="The length of action sequences (action_horizon).",
    )
    args = parser.parse_args()
    config_path = Path(args.config_path)
    if not config_path.exists():
        logging.error(f"Configuration file does not exist: {config_path}")
        return

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    for arm_name, arm_config in config.get("data_groups", {}).items():
        for dataset_name, dataset_config in arm_config.items():
            path_str = dataset_config.get("path")
            if not path_str:
                logging.warning(
                    f"Dataset '{arm_name}/{dataset_name}' has no path configured, skipping."
                )
                continue

            dataset_path = Path(path_str)
            use_delta = dataset_config.get("use_delta_action", False)

            compute_normstats(
                dataset_path,
                use_delta_actions=use_delta,
                action_horizon=args.action_horizon,
                dataset_config=dataset_config,
            )


if __name__ == "__main__":
    main()
