#!/usr/bin/env python

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

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


@dataclass(frozen=True)
class ACPIndicatorStats:
    indicator_field: str
    total_count: int
    positive_count: int
    positive_ratio: float
    invalid_count: int
    source: str


def _stats_entry_to_scalar(stats_entry: dict[str, Any], key: str) -> float | None:
    if key not in stats_entry:
        return None
    values = np.asarray(stats_entry[key]).reshape(-1)
    if values.size == 0:
        return None
    return float(values[0])


def _from_meta_stats(dataset: Any, indicator_field: str) -> ACPIndicatorStats | None:
    meta = getattr(dataset, "meta", None)
    if meta is None:
        return None
    stats = getattr(meta, "stats", None)
    if not isinstance(stats, dict):
        return None
    if indicator_field not in stats:
        return None

    field_stats = stats[indicator_field]
    if not isinstance(field_stats, dict):
        return None

    ratio = _stats_entry_to_scalar(field_stats, "mean")
    if ratio is None:
        return None

    count_raw = _stats_entry_to_scalar(field_stats, "count")
    total_count = int(round(count_raw)) if count_raw is not None else -1
    positive_count = int(round(ratio * total_count)) if total_count >= 0 else -1

    return ACPIndicatorStats(
        indicator_field=indicator_field,
        total_count=total_count,
        positive_count=positive_count,
        positive_ratio=ratio,
        invalid_count=-1,
        source="meta.stats",
    )


def _from_hf_dataset_scan(dataset: Any, indicator_field: str) -> ACPIndicatorStats | None:
    hf_dataset = getattr(dataset, "hf_dataset", None)
    if hf_dataset is None:
        return None

    column_names = getattr(hf_dataset, "column_names", None)
    if column_names is not None and indicator_field not in column_names:
        return None

    column_values = hf_dataset[indicator_field]

    total_count = 0
    positive_count = 0
    invalid_count = 0
    for value in column_values:
        values = torch.as_tensor(value).reshape(-1)
        total_count += int(values.numel())
        positive_count += int((values == 1).sum().item())
        invalid_count += int((~((values == 0) | (values == 1))).sum().item())

    if total_count == 0:
        return None

    return ACPIndicatorStats(
        indicator_field=indicator_field,
        total_count=total_count,
        positive_count=positive_count,
        positive_ratio=positive_count / total_count,
        invalid_count=invalid_count,
        source="hf_dataset_scan",
    )


def compute_acp_indicator_stats(dataset: Any, indicator_field: str) -> ACPIndicatorStats | None:
    """Compute ACP indicator ratio from dataset metadata or by scanning the indicator column."""
    stats_from_meta = _from_meta_stats(dataset, indicator_field)
    if stats_from_meta is not None:
        return stats_from_meta

    return _from_hf_dataset_scan(dataset, indicator_field)
