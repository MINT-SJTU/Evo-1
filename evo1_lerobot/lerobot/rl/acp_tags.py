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

"""Shared ACP prompt tags used by both training and inference."""

ACP_TAG_KEY = "Advantage"
ACP_POSITIVE_VALUE = "positive"
ACP_NEGATIVE_VALUE = "negative"

ACP_POSITIVE_TAG = f"{ACP_TAG_KEY}: {ACP_POSITIVE_VALUE}"
ACP_NEGATIVE_TAG = f"{ACP_TAG_KEY}: {ACP_NEGATIVE_VALUE}"


def build_acp_tagged_task(task: str | None, is_positive: bool) -> str:
    tag = ACP_POSITIVE_TAG if is_positive else ACP_NEGATIVE_TAG
    base_task = task or ""
    if not base_task:
        return tag
    return f"{base_task}\n{tag}"
