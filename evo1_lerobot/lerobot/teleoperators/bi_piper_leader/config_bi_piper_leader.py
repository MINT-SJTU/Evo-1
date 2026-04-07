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

from lerobot.teleoperators.piper_leader import PiperLeaderConfigBase, PiperXLeaderConfigBase

from ..config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("bi_piper_leader")
@dataclass
class BiPiperLeaderConfig(TeleoperatorConfig):
    """Configuration class for bimanual PiPER leader teleoperators."""

    left_arm_config: PiperLeaderConfigBase
    right_arm_config: PiperLeaderConfigBase
    process_isolation: bool = True


@TeleoperatorConfig.register_subclass("bi_piperx_leader")
@dataclass
class BiPiperXLeaderConfig(TeleoperatorConfig):
    """Configuration class for bimanual PiPER-X leader teleoperators."""

    left_arm_config: PiperXLeaderConfigBase
    right_arm_config: PiperXLeaderConfigBase
    process_isolation: bool = True
