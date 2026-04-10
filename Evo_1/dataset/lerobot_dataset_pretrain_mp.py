import json
import logging
import random
from collections import OrderedDict, defaultdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Union

import numpy as np
import pandas as pd
import pickle
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.functional import InterpolationMode
from tqdm.auto import tqdm

from dataset.dataset_process_suite import get_suite
from dataset.compute_normstats import compute_normstats as compute_normstats_regular
from dataset.compute_normstats_streaming import compute_normstats as compute_normstats_streaming


CACHE_SCHEMA_VERSION = 2
MANIFEST_FILE = "manifest.json"
INDEX_FILE = "index.pkl"
SHARD_PREFIX = "data_"
SHARD_SUFFIX = ".bin"


class NormalizationType(str, Enum):
    NORMAL = "normal"
    BOUNDS = "bounds"
    BOUNDS_Q99 = "bounds_q99"


class _VideoDecoderLRU:
    """Reusable decoder cache to avoid re-opening videos for every sample."""

    def __init__(self, backend: str, backend_kwargs: Optional[Dict[str, Any]] = None, max_size: int = 16):
        self.backend = backend
        self.backend_kwargs = backend_kwargs or {}
        self.max_size = max(1, int(max_size))
        self._cache: "OrderedDict[str, Any]" = OrderedDict()

    def _evict_if_needed(self):
        while len(self._cache) > self.max_size:
            _, handle = self._cache.popitem(last=False)
            try:
                handle.close()
            except Exception:
                pass

    def close_all(self):
        for _, handle in self._cache.items():
            try:
                handle.close()
            except Exception:
                pass
        self._cache.clear()

    def _get_decord_reader(self, path: str):
        import decord

        reader = self._cache.get(path)
        if reader is not None:
            self._cache.move_to_end(path)
            return reader

        ctx = self.backend_kwargs.get("ctx", "cpu")
        if ctx == "cpu":
            ctx = decord.cpu(0)
        elif ctx == "gpu":
            ctx = decord.gpu(0)
        reader = decord.VideoReader(path, ctx=ctx)
        self._cache[path] = reader
        self._evict_if_needed()
        return reader

    def _get_pyav_container(self, path: str):
        import av

        container = self._cache.get(path)
        if container is not None:
            self._cache.move_to_end(path)
            return container

        container = av.open(path)
        self._cache[path] = container
        self._evict_if_needed()
        return container

    def _get_torchcodec_decoder(self, path: str):
        import torchcodec

        decoder = self._cache.get(path)
        if decoder is not None:
            self._cache.move_to_end(path)
            return decoder

        decoder = torchcodec.VideoDecoder(path)
        self._cache[path] = decoder
        self._evict_if_needed()
        return decoder

    def decode_single_frame(self, path: str, timestamp: float) -> Image.Image:
        if self.backend == "torchcodec":
            decoder = self._get_torchcodec_decoder(path)
            fps = float(decoder.metadata.average_fps)
            if fps is None or np.isnan(fps):
                raise RuntimeError(f"Invalid FPS while decoding {path}")
            frame_idx = int(max(0.0, float(timestamp)) * fps)
            frame_idx = max(0, min(frame_idx, len(decoder) - 1))
            
            frame_tensor = decoder[frame_idx].data
            frame_np = frame_tensor.permute(1, 2, 0).cpu().numpy()
            return Image.fromarray(frame_np, mode="RGB")

        if self.backend == "decord":
            reader = self._get_decord_reader(path)
            fps = reader.get_avg_fps()
            if fps is None or np.isnan(fps):
                raise RuntimeError(f"Invalid FPS while decoding {path}")
            frame_idx = int(max(0.0, float(timestamp)) * float(fps))
            frame_idx = max(0, min(frame_idx, len(reader) - 1))
            frame = reader.get_batch([frame_idx]).asnumpy()[0]
            return Image.fromarray(frame)

        if self.backend == "av":
            container = self._get_pyav_container(path)
            stream = container.streams.video[0]
            container.seek(0, stream=stream)
            last_frame = None
            for frame in container.decode(video=0):
                last_frame = frame
                if frame.time is not None and frame.time >= timestamp:
                    return Image.fromarray(frame.to_ndarray(format="rgb24"))
            if last_frame is None:
                raise RuntimeError(f"No frame decoded from {path}")
            return Image.fromarray(last_frame.to_ndarray(format="rgb24"))

        raise NotImplementedError(f"Unsupported video backend: {self.backend}")


class LeRobotDataset(Dataset):
    def __init__(
        self,
        config: Dict[str, Any],
        image_size: int = 448,
        max_samples_per_file: Union[int, None] = None,
        video_backend: str = "av",
        action_horizon: int = 50,
        video_backend_kwargs: Dict[str, Any] = None,
        binarize_gripper: bool = False,
        cache_dir: Union[str, Path] = None,
        use_augmentation: bool = False,
    ):
        self.config = config
        self.image_size = image_size
        self.max_samples_per_file = max_samples_per_file
        self.binarize_gripper = binarize_gripper
        self.use_augmentation = use_augmentation
        self.action_horizon = action_horizon
        self.video_backend = video_backend
        self.video_backend_kwargs = video_backend_kwargs or {}
        self.normalization_type = self.config.get("normalization_type", NormalizationType.BOUNDS.value)

        self.max_action_dim = config["max_action_dim"]
        self.max_state_dim = config["max_state_dim"]
        self.max_views = config["max_views"]

        sorted_datasets = sorted(self.config["data_groups"].keys())
        self.arm_to_embodiment_id = {key: i for i, key in enumerate(sorted_datasets)}

        default_cache_root = Path(".") / "dataset" / "dataset_cache"
        self.cache_dir = Path(cache_dir) if cache_dir is not None else default_cache_root
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        export_key_raw = str(self.config.get("datasets_manifest", "datasets_manifest.pkl"))
        self.export_key = Path(export_key_raw).stem
        self.manifest_root = self.cache_dir / "manifest" / self.export_key
        self.manifest_root.mkdir(parents=True, exist_ok=True)

        self.data_cache_root = self.cache_dir / "datagroup_cache" / self.export_key
        self.data_cache_root.mkdir(parents=True, exist_ok=True)

        self._data_file_handles: Dict[str, Any] = {}
        self._decoder_cache = _VideoDecoderLRU(
            backend=self.video_backend,
            backend_kwargs=self.video_backend_kwargs,
            max_size=int(self.config.get("decoder_cache_size", 16)),
        )

        self.dataset_entries: List[Dict[str, Any]] = []
        self.arm2stats_dict: Dict[str, Dict[str, Any]] = {}
        self._view_order_map: Dict[tuple, List[str]] = {}
        self._index: List[Dict[str, Union[str, int]]] = []
        self._manifest: Dict[str, Any] = {}

        self._load_metadata()
        
        try:
            from accelerate import PartialState
            state = PartialState()
            if state.is_main_process:
                self._ensure_cache_exported()
            state.wait_for_everyone() 
        except ImportError:
            logging.warning("Accelerate not found; skipping distributed cache export synchronization.")
            self._ensure_cache_exported()
            
        self._load_runtime_index()

        self.basic_transform = T.Compose(
            [
                T.Resize((self.image_size, self.image_size), interpolation=InterpolationMode.BICUBIC),
                T.ToTensor(),
            ]
        )
        self.aug_transform = T.Compose(
            [
                T.RandomResizedCrop(self.image_size, scale=(0.95, 1.0), interpolation=InterpolationMode.BICUBIC),
                T.RandomRotation(degrees=(-5, 5), interpolation=InterpolationMode.BICUBIC),
                T.ColorJitter(brightness=0.3, contrast=0.4, saturation=0.5, hue=0.08),
                T.ToTensor(),
            ]
        )

    def __del__(self):
        self._close_runtime_resources()

    def _close_runtime_resources(self):
        for _, f in self._data_file_handles.items():
            try:
                f.close()
            except Exception:
                pass
        self._data_file_handles.clear()
        self._decoder_cache.close_all()

    def _load_metadata(self):
        self.dataset_entries = []
        self.arm2stats_dict = {}
        self._view_order_map = {}

        for arm_name, arm_config in self.config["data_groups"].items():
            self.arm2stats_dict[arm_name] = {}
            for dataset_name, dataset_config in arm_config.items():
                dataset_path = Path(dataset_config["path"])
                tasks_path = dataset_path / "meta" / "tasks.jsonl"
                if not tasks_path.exists():
                    raise FileNotFoundError(f"tasks file not found: {tasks_path}")
                task_records = pd.read_json(tasks_path, lines=True).to_dict("records")
                task_mapping = {
                    obj["task_index"]: obj["task"]
                    for obj in task_records
                    if "task_index" in obj and "task" in obj
                }

                stats_path = dataset_path / "meta" / "episodes_stats.jsonl"
                stats_path_after_compute = dataset_path / "meta" / "stats.json"
                use_delta_action = dataset_config.get("use_delta_action", False)

                if stats_path_after_compute.exists():
                    with open(stats_path_after_compute, "r", encoding="utf-8") as f:
                        stats = json.load(f)
                else:
                    logging.info(f"Computing norm stats for {dataset_path}...")
                    try:
                        compute_normstats_regular(
                            dataset_path, 
                            use_delta_actions=use_delta_action, 
                            action_horizon=self.action_horizon, 
                            dataset_config=dataset_config
                        )
                    except (MemoryError, Exception) as e:
                        logging.warning(f"OOM or error during regular normstats compute ({e}). Falling back to streaming.")
                        compute_normstats_streaming(
                            dataset_path, 
                            use_delta_actions=use_delta_action, 
                            action_horizon=self.action_horizon, 
                            dataset_config=dataset_config
                        )
                    
                    if stats_path_after_compute.exists():
                        with open(stats_path_after_compute, "r", encoding="utf-8") as f:
                            stats = json.load(f)
                    else:
                        raise FileNotFoundError(f"normalization stats file not found after compute: {stats_path_after_compute}")

                self.arm2stats_dict[arm_name][dataset_name] = stats

                parquet_files = sorted((dataset_path / "data").glob("*/*.parquet"))
                if not parquet_files:
                    logging.warning("No parquet files found under %s", dataset_path / "data")

                view_map = dataset_config.get("view_map")
                if not view_map:
                    default_keys = ["image_1", "image_2", "image_3"]
                    view_map = {key: f"observation.images.{key}" for key in default_keys}
                self._view_order_map[(arm_name, dataset_name)] = list(view_map.keys())

                self.dataset_entries.append(
                    {
                        "arm_name": arm_name,
                        "dataset_name": dataset_name,
                        "dataset_path": dataset_path,
                        "task_mapping": task_mapping,
                        "view_map": view_map,
                        "use_delta_action": use_delta_action,
                        "suite_name": dataset_config.get("process_suite", "default"),
                        "suite_config": dataset_config.get("suite_config", {}),
                        "parquet_files": parquet_files,
                    }
                )

    def _iter_window_records(self, entry: Dict[str, Any], parquet_path: Path) -> Iterator[Dict[str, Any]]:
        suite = get_suite(entry["suite_name"], entry["suite_config"])
        df = pd.read_parquet(parquet_path)
        if len(df) == 0:
            return

        view_order = list(entry["view_map"].keys())

        last_row = df.iloc[-1:]
        padding_rows = pd.concat([last_row] * self.action_horizon, ignore_index=True)
        df = pd.concat([df, padding_rows], ignore_index=True)
        if self.max_samples_per_file is not None:
            df = df.head(self.max_samples_per_file)

        for i in range(len(df) - self.action_horizon + 1):
            sub_df = df.iloc[i : i + self.action_horizon]
            processed = suite.process(sub_df, use_delta_action=entry["use_delta_action"])

            video_paths = {}
            base_video_path = entry["dataset_path"] / "videos" / parquet_path.parent.name
            for view_key, view_folder in entry["view_map"].items():
                full_path = base_video_path / view_folder / f"{parquet_path.stem}.mp4"
                if full_path.exists():
                    video_paths[view_key] = str(full_path)

            task_index = sub_df.iloc[0].get("task_index", None)
            prompt = entry["task_mapping"].get(task_index, "")

            yield {
                "arm_key": entry["arm_name"],
                "dataset_key": entry["dataset_name"],
                "prompt": prompt,
                "state": processed.state,
                "action": processed.actions,
                "timestamp": sub_df.iloc[0].get("timestamp", None),
                "video_paths": video_paths,
                "view_order": view_order,
            }

    def _write_json_atomic(self, path: Path, payload: Dict[str, Any]):
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        tmp_path.replace(path)

    def _write_pickle_atomic(self, path: Path, payload: Any):
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        with open(tmp_path, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        tmp_path.replace(path)

    def _ensure_cache_exported(self):
        manifest_path = self.manifest_root / MANIFEST_FILE
        index_path = self.manifest_root / INDEX_FILE
        if manifest_path.exists() and index_path.exists():
            return

        logging.info("Exporting dataset cache into split layout under %s", self.cache_dir)
        shard_target_size = int(self.config.get("cache_shard_size_bytes", 256 * 1024 * 1024))
        index: List[Dict[str, Union[str, int]]] = []
        data_files: List[str] = []
        sample_count = 0

        total_parquet = sum(len(e["parquet_files"]) for e in self.dataset_entries)
        with tqdm(total=total_parquet, desc="Export parquet windows") as pbar:
            for entry in self.dataset_entries:
                arm_name = entry["arm_name"]
                dataset_name = entry["dataset_name"]
                dataset_cache_dir = self.data_cache_root / arm_name / dataset_name
                dataset_cache_dir.mkdir(parents=True, exist_ok=True)

                shard_id = 0
                shard_size = 0
                shard_fh = None
                shard_rel = None

                def open_new_shard() -> Any:
                    nonlocal shard_id, shard_size, shard_rel
                    shard_name = f"{SHARD_PREFIX}{shard_id:05d}{SHARD_SUFFIX}"
                    shard_path = dataset_cache_dir / shard_name
                    shard_rel = str(shard_path.relative_to(self.data_cache_root)).replace("\\\\", "/")
                    if shard_rel not in data_files:
                        data_files.append(shard_rel)
                    shard_id += 1
                    shard_size = 0
                    return open(shard_path, "wb")

                try:
                    shard_fh = open_new_shard()
                    for parquet_path in entry["parquet_files"]:
                        for record in self._iter_window_records(entry, parquet_path):
                            blob = pickle.dumps(record, protocol=pickle.HIGHEST_PROTOCOL)
                            if shard_size > 0 and shard_size + len(blob) > shard_target_size:
                                shard_fh.close()
                                shard_fh = open_new_shard()
                            offset = shard_size
                            shard_fh.write(blob)
                            shard_size += len(blob)
                            index.append(
                                {
                                    "file_rel": shard_rel,
                                    "offset": offset,
                                    "size": len(blob),
                                }
                            )
                            sample_count += 1
                        pbar.update(1)
                finally:
                    if shard_fh is not None:
                        shard_fh.close()

        if sample_count == 0:
            raise RuntimeError("Cache export produced zero samples; please check dataset paths/config.")

        manifest = {
            "schema_version": CACHE_SCHEMA_VERSION,
            "export_key": self.export_key,
            "layout": "manifest+datagroup_cache",
            "manifest_root": str(self.manifest_root),
            "data_cache_root": str(self.data_cache_root),
            "sample_count": sample_count,
            "action_horizon": self.action_horizon,
            "max_state_dim": self.max_state_dim,
            "max_action_dim": self.max_action_dim,
            "max_views": self.max_views,
            "normalization_type": self.normalization_type,
            "media_policy": "source_video_reference",
            "arm_to_embodiment_id": self.arm_to_embodiment_id,
            "data_files": data_files,
            "index_file": INDEX_FILE,
        }

        self._write_pickle_atomic(index_path, index)
        self._write_json_atomic(manifest_path, manifest)

    def _load_runtime_index(self):
        manifest_path = self.manifest_root / MANIFEST_FILE
        index_path = self.manifest_root / INDEX_FILE
        if not manifest_path.exists() or not index_path.exists():
            raise RuntimeError(f"Cache files missing under {self.manifest_root}")

        with open(manifest_path, "r", encoding="utf-8") as f:
            self._manifest = json.load(f)
        if self._manifest.get("schema_version") != CACHE_SCHEMA_VERSION:
            raise RuntimeError(
                f"Unsupported cache schema {self._manifest.get('schema_version')} for {manifest_path}"
            )

        with open(index_path, "rb") as f:
            self._index = pickle.load(f)

        for rel_path in self._manifest["data_files"]:
            data_path = self.data_cache_root / rel_path
            if not data_path.exists():
                raise RuntimeError(f"Data shard missing: {data_path}")

        expected = int(self._manifest["sample_count"])
        if expected != len(self._index):
            raise RuntimeError(f"Manifest/index mismatch: sample_count={expected}, index={len(self._index)}")

    def _read_record(self, idx: int) -> Dict[str, Any]:
        loc = self._index[idx]
        file_rel = str(loc["file_rel"])
        offset = int(loc["offset"])
        size = int(loc["size"])

        if file_rel not in self._data_file_handles:
            self._data_file_handles[file_rel] = open(self.data_cache_root / file_rel, "rb")
        fh = self._data_file_handles[file_rel]

        fh.seek(offset)
        blob = fh.read(size)
        if len(blob) != size:
            raise RuntimeError(f"Corrupted shard read at index {idx}: expected {size} bytes, got {len(blob)}")
        try:
            return pickle.loads(blob)
        except Exception as e:
            raise RuntimeError(f"Failed to deserialize sample at index {idx}: {e}") from e

    def _load_video_frame(
        self,
        video_paths: Dict[str, str],
        timestamp: float,
        view_order: Optional[List[str]] = None,
    ) -> List[Optional[Image.Image]]:
        ordered_views = view_order if view_order is not None else list(video_paths.keys())
        if not ordered_views:
            raise RuntimeError("Sample has empty view ordering and video_paths")

        path_to_views = defaultdict(list)
        for view_key in ordered_views:
            path = video_paths.get(view_key)
            if path:
                path_to_views[path].append(view_key)

        decoded_by_path: Dict[str, Image.Image] = {}
        for path in path_to_views:
            path_obj = Path(path)
            if not path_obj.exists():
                raise FileNotFoundError(f"video file not found: {path}")
            decoded_by_path[path] = self._decoder_cache.decode_single_frame(path, float(timestamp))

        frames: List[Optional[Image.Image]] = []
        for view_key in ordered_views:
            path = video_paths.get(view_key)
            frames.append(decoded_by_path[path] if path in decoded_by_path else None)
        return frames

    def _pad_last_dim(self, source_tensor: torch.Tensor, max_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
        source_dim = source_tensor.shape[-1]
        if source_dim > max_dim:
            raise ValueError(f"Source dim {source_dim} exceeds configured max dim {max_dim}")

        padded_shape = (*source_tensor.shape[:-1], max_dim) if source_tensor.dim() > 1 else (max_dim,)
        padded_tensor = torch.zeros(padded_shape, dtype=source_tensor.dtype, device=source_tensor.device)
        mask = torch.zeros(padded_shape, dtype=torch.bool, device=source_tensor.device)
        padded_tensor[..., :source_dim] = source_tensor
        mask[..., :source_dim] = True
        return padded_tensor, mask

    def _normalize_tensor(self, tensor, stats_dict, stats_name, normalization_type, apply_zero_mask=False):
        eps = 1e-8
        device = tensor.device
        input_dim = tensor.shape[-1]

        def _stat_tensor(key):
            values = stats_dict.get(key)
            if values is None:
                return None
            stat_t = torch.tensor(values, dtype=torch.float32, device=device)
            if stat_t.shape[-1] > input_dim:
                stat_t = stat_t[..., :input_dim]
            elif stat_t.shape[-1] < input_dim:
                raise ValueError(
                    f"Stats dimension ({stat_t.shape[-1]}) is smaller than tensor dimension ({input_dim}) "
                    f"for {stats_name}.{key}"
                )
            return stat_t

        if normalization_type == NormalizationType.NORMAL:
            mean = _stat_tensor("mean")
            std = _stat_tensor("std")
            if mean is None or std is None:
                raise ValueError(f"Missing mean/std stats for {stats_name}")
            return (tensor - mean) / (std + eps)

        if normalization_type == NormalizationType.BOUNDS:
            low = _stat_tensor("min")
            high = _stat_tensor("max")
            if low is None or high is None:
                raise ValueError(f"Missing min/max stats for {stats_name}")
        elif normalization_type == NormalizationType.BOUNDS_Q99:
            low = _stat_tensor("q01")
            high = _stat_tensor("q99")
            if low is None or high is None:
                logging.warning("Missing q01/q99 for %s; fallback to min/max", stats_name)
                low = _stat_tensor("min")
                high = _stat_tensor("max")
            if low is None or high is None:
                raise ValueError(f"Missing q01/q99 and min/max for {stats_name}")
        else:
            raise ValueError(f"Unsupported normalization type: {normalization_type}")

        diff = high - low
        normalized = 2 * (tensor - low) / (diff + eps) - 1
        normalized = torch.clamp(normalized, -1.0, 1.0)
        if apply_zero_mask:
            zeros_mask = diff.abs() < eps
            while zeros_mask.dim() < normalized.dim():
                zeros_mask = zeros_mask.unsqueeze(0)
            normalized = torch.where(zeros_mask, torch.zeros_like(normalized), normalized)
        return normalized

    def _prepare_state_action(self, record: Dict[str, Any]):
        arm_key = record["arm_key"]
        dataset_key = record["dataset_key"]
        if record["state"] is None:
            raise ValueError("Missing observation.state in cache record")
        if record["action"] is None:
            raise ValueError("Missing action in cache record")

        stats = self.arm2stats_dict.get(arm_key, {}).get(dataset_key)
        if stats is None:
            raise KeyError(f"Normalization stats not found for arm={arm_key} dataset={dataset_key}")

        state = torch.tensor(record["state"], dtype=torch.float32)
        state = self._normalize_tensor(
            tensor=state,
            stats_dict=stats["observation.state"],
            stats_name="observation.state",
            normalization_type=self.normalization_type,
            apply_zero_mask=False,
        )
        state_padded, state_mask = self._pad_last_dim(state, self.max_state_dim)

        action_np = np.asarray(record["action"], dtype=np.float32)
        action = torch.from_numpy(action_np)
        action = self._normalize_tensor(
            tensor=action,
            stats_dict=stats["action"],
            stats_name="action",
            normalization_type=self.normalization_type,
            apply_zero_mask=True,
        )
        action_padded, action_mask = self._pad_last_dim(action, self.max_action_dim)
        return state_padded, state_mask, action_padded, action_mask

    def __len__(self):
        return len(self._index)

    def __getitem__(self, idx):
        try:
            record = self._read_record(idx)
        except Exception as e:
            logging.info("cannot load cache record at idx=%s: %s", idx, e)
            return self[random.randint(0, len(self._index) - 1)]

        arm_key = record["arm_key"]
        dataset_key = record["dataset_key"]
        embodiment_id = self.arm_to_embodiment_id[arm_key]
        view_order = record.get("view_order")
        if view_order is None:
            view_order = self._view_order_map.get((arm_key, dataset_key))

        try:
            frames = self._load_video_frame(
                record["video_paths"],
                record["timestamp"],
                view_order=view_order,
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to decode video for sample idx={idx}, arm={arm_key}, dataset={dataset_key}: {e}"
            ) from e

        # use only successfully decoded views, and build a contiguous mask.
        valid_frames = [frame for frame in frames if frame is not None]
        images: List[torch.Tensor] = []
        if self.use_augmentation:
            images = [
                self.aug_transform(img) if random.random() < 0.5 else self.basic_transform(img)
                for img in valid_frames
            ]
        else:
            images = [self.basic_transform(img) for img in valid_frames]

        num_real_views = len(images)
        image_mask = torch.zeros(self.max_views, dtype=torch.bool)
        image_mask[:num_real_views] = True

        while len(images) < self.max_views:
            if images:
                images.append(torch.zeros_like(images[0]))
            else:
                images.append(torch.zeros(3, self.image_size, self.image_size))
        images = torch.stack(images)

        state_padded, state_mask, action_padded, action_mask = self._prepare_state_action(record)

        return {
            "images": images,
            "image_mask": image_mask,
            "prompt": record.get("prompt") or "",
            "state": state_padded.to(dtype=torch.bfloat16),
            "state_mask": state_mask,
            "action": action_padded.to(dtype=torch.bfloat16),
            "action_mask": action_mask,
            "embodiment_id": torch.tensor(embodiment_id, dtype=torch.long),
        }
