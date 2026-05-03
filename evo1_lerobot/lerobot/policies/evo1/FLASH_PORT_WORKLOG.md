# Evo1 Flash Port Worklog

## Objective

补齐 `origin/evo1-flash` 中对训练稳定性和吞吐有价值、且适合映射到 `evo1_lerobot` 的改动。

## Completion Criteria

- 明确列出 `flash` 中缺失且应迁移的改动项
- 在 `evo1_lerobot` 中完成对应代码迁移
- 通过 reviewer 复核迁移边界和实现
- 在 4x3090 环境完成改动前后训练吞吐对比
- 完成至少一轮一致性检查，确认 loss/forward 行为未出现明显异常偏移

## Assumptions

- `origin/evo1-flash` 是基于原生 `Evo_1/` 的训练优化分支，不应把其原生训练脚本和自定义 dataset cache 体系直接搬进 LeRobot 训练栈
- `evo1_lerobot` 继续以 `lerobot.scripts.lerobot_train` 为唯一训练入口
- 当前最值得迁移的是模型侧和前向路径优化，而不是原生训练基础设施

## Candidate Changes From `evo1-flash`

### Port

- `InternVL3` 批量图像 / 批量 prompt 处理，替代逐样本 embedder 调用
- `InternVL3` 的 `flash_attn` 可用性检测与回退
- `InternVL3` 更完整的 gradient checkpointing 配置
- `FlowmatchingActionHead` 的 dtype 对齐、mask 展开、time index 边界处理
- `CategorySpecificLinear` / action encoder 的稳定性修正
- `EVO1` 分支级 finetune 能力：
  - `finetune_language_model`
  - `finetune_vision_model`

### Do Not Port Directly

- 原生 `Evo_1/scripts/train.py` 的训练主循环
- 原生 `Evo_1/config.py`
- 原生 checkpoint 保存/恢复格式
- 原生 `lerobot_dataset_pretrain_mp.py` 的 cache/manifest/shard/process suite 体系

## Risks

- 批量化 embedder 后，token 对齐或 image mask 处理可能引入静默行为差异
- gradient checkpointing 设置不当可能导致 InternVL3 远程代码路径兼容性问题
- flow matching 的 dtype/mask 修正可能影响已有 checkpoint 的推理边界

## Validation Plan

- 静态检查：相关模块 import / config 路径正确
- 单步训练 smoke：policy forward + backward 能运行
- 4x3090 前后对比：
  - step time
  - samples/sec
  - 显存占用
- 一致性检查：
  - 相同 batch 上 loss 是否处于合理接近范围
  - action / velocity tensor shape 与 mask 对齐

## Review Questions

- 迁移边界是否合理，是否遗漏了高价值的 `flash` 改动
- 哪些改动会和 LeRobot 训练栈冲突，不应迁入
- 是否需要把 fused AdamW / dataloader 参数也纳入本轮

## Reviewer Cycle 1

### Milestone Reviewed

- `flash -> evo1_lerobot` 迁移范围划定

### Reviewer Findings

- 当前“应迁移 / 不应直接迁移”的划分总体正确
- 本轮应优先只做三块：
  - `InternVL3` 批量化 + checkpointing/fallback 处理
  - `flow_matching` 稳定性修正
  - `configuration_evo1.py` 配置面补齐
- 不建议本轮同时纳入 `fused AdamW` 与 dataloader 参数调优，否则性能归因会混淆
- 需要把行为一致性检查设计得更明确，不能只看 samples/sec

### Response / Scope Tightening

- 本轮明确排除：
  - fused AdamW
  - dataloader 参数可配置化
  - 原生训练脚本与原生 dataset cache
- 本轮明确纳入：
  - batched `InternVL3` embedding 路径
  - `flash_attn` fallback
  - 更完整的 gradient checkpointing 处理
  - `flow_matching` 的 dtype / mask / shape 稳定性修正
  - `stage1/stage2` 对 branch-level finetune 的配置语义

## Experiment Design

- baseline commit:
  - 当前工作树在迁移前的 `evo1_lerobot` 实现
- target commit:
  - 完成本轮三块核心迁移后的实现
- fixed controls:
  - 同一数据集
  - 同一 accelerate 配置
  - 同一 batch size / num_workers / precision
  - 不引入 fused AdamW
- metrics:
  - `sec/step`
  - `samples/sec`
  - GPU memory allocated/reserved
- consistency checks:
  - 相同 batch 上 `fused_tokens` shape 一致
  - 相同 batch 上 `pred_velocity` / `loss` 数值处于可解释接近范围
  - `get_action()` 在 `horizon=1` 与 `horizon>1` 下的 mask / shape 自洽

## Completed Validation

- `py_compile` passed for:
  - `configuration_evo1.py`
  - `evo1_model.py`
  - `internvl3_embedder.py`
  - `flow_matching.py`
  - `modeling_evo1.py`
- `flow_matching` smoke passed on local `lerobot` env:
  - `horizon>1`
  - `horizon=1`
- During smoke, discovered and fixed a real bug:
  - `horizon=1` training path produced a 4D attention query due to an extra unsqueeze in `noise_seq`
- `configuration_evo1.py` stage semantics smoke passed:
  - `stage1 -> finetune_vlm=False, finetune_language_model=False, finetune_vision_model=False, finetune_action_head=True`
  - `stage2 default -> True, True, True, True`
  - `stage2 language only -> True, True, False, True`
- `InternVL3` batched entry smoke passed with mocked tokenizer/model:
  - 2 samples
  - 2 image views per sample
  - non-trivial image mask path
  - verified both `return_cls_only=False` and `return_cls_only=True`

## Current Blockers

- Current machine is not the 4x3090 training host
- `nvidia-smi` is unavailable locally, so throughput validation on 4x3090 cannot be executed from this machine without remote host access

## Reviewer Cycle 2

### Milestone Reviewed

- 第一轮 flash 迁移实现

### Reviewer Findings

- `stage2` 语义会覆盖显式关闭 branch 的配置
- policy 层 `chunk_size=1` 时 2D `action_mask` 仍有兼容问题
- branch-level finetune 当前只是部分映射，不应表述为完整 VLM branch 控制
- batched embedder 的 tensor 图像预处理路径与旧实现不完全等价
- 仍缺真实模型级对照和 4x3090 实验

### Fixes Applied

- 将 `finetune_vlm` / `finetune_language_model` / `finetune_vision_model` / `finetune_action_head`
  改为 `bool | None`，用 `None` 表示“未显式指定”，避免 stage 语义覆盖显式关闭意图
- 修复 `modeling_evo1._prepare_actions()` 中 `chunk_size=1` + 2D `action_mask` 的 `unsqueeze` 维度错误
- `internvl3_embedder.py` 中将 tensor 图像输入预处理尽量拉回旧的 PIL 路径语义
- `enable_gradient_checkpointing=False` 时对 language model 也做对称关闭处理

### Additional Validation

- `stage2-explicit-off` smoke passed:
  - `finetune_vlm=False`
  - `finetune_language_model=False`
  - `finetune_vision_model=False`
  - `finetune_action_head=False`
- policy-level helper smoke passed:
  - `_prepare_actions()` now supports `chunk_size=1` + 2D `action_mask`
  - `_compute_fused_tokens()` verified single batched call into `get_vl_embeddings()`
- mocked InternVL3 batched smoke re-passed after preprocessing fix

### Residual Risks

- branch-level finetune 仍是基于当前已知 `language_model` / `vision_model` / `mlp1` 的部分映射
- batched embedder 还没有完成真实 InternVL3 权重下的 per-sample vs batched 数值对照
- 4x3090 吞吐和一致性实验仍待远端机器执行

## Final Guard Fixes

- `configuration_evo1.py` 新增 finetune 配置一致性约束：
  - 如果 `finetune_vlm` 与 `(finetune_language_model or finetune_vision_model)` 不一致，直接抛出 `ValueError`
- 清理 `internvl3_embedder.py` 中不再被主路径使用的 tensor 预处理辅助函数，避免维护误导

## Remote Experiment Status

- Found a local SSH alias that looks like the 4x3090 host: `3090_4`
- Read-only connectivity probe failed:
  - `ssh -o BatchMode=yes 3090_4 'hostname ...'`
  - observed result: `Connection closed by 10.60.18.143 port 22`
- Therefore 4x3090 throughput / consistency experiment is currently blocked by remote host access, not by local code readiness
