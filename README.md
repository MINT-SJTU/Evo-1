# Evo1 LeRobot Branch

This branch is a LeRobot-adapted Evo1 implementation intended to live as the `evo1-lerobot` branch of the Evo-1 repository.

Compared with the original Evo1 codebase, this version moves Evo1 into the LeRobot training stack and benefits from a more structured training framework. In practice, this version has shown noticeably better training throughput than the original monolithic Evo1 training script.

- code root: `./evo1_lerobot/lerobot`
- launch entry: `python -m accelerate.commands.launch -m lerobot.scripts.lerobot_train`

`lerobot` is not listed in `requirements.txt` because it is included in this repository as source code under [evo1_lerobot/lerobot](./evo1_lerobot/lerobot).

## Main Differences from Native Evo1

- Training is integrated into the LeRobot framework instead of using one large standalone training script.
- Dataset loading follows the LeRobot dataset pipeline and uses LeRobot v3-style dataset layout.
- Checkpoints follow the LeRobot policy format rather than the original Evo1 checkpoint layout.
- Stage handoff uses `--policy.path=<...>/pretrained_model` together with `--resume_pretrain=true`.

## Environment

Use [evo1_lerobot/requirements.txt](./evo1_lerobot/requirements.txt).

Validated package versions:

- `torch==2.5.1+cu124`
- `torchvision==0.20.1+cu124`
- `flash-attn==2.8.3`
- `torchcodec==0.1.1`
- `transformers==4.53.3`

Install:

```bash
conda create -y -n evo1-lerobot python=3.10 pip
conda activate evo1-lerobot
python -m pip install --upgrade pip
python -m pip install -r evo1_lerobot/requirements.txt
```

For restricted machines, pre-download the wheels and install from a local wheelhouse instead of relying on public indexes.

`flash-attn` is part of the validated setup and should be installed.

## TorchCodec Runtime

`torchcodec` also needs a working FFmpeg runtime. In the validated restricted-machine run, training used:

```bash
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
export LD_LIBRARY_PATH=/path/to/ffmpeg_runtime/lib:/path/to/libiconv/lib:${LD_LIBRARY_PATH:-}
```

If `torchcodec` is not configured correctly, training may still work with `pyav`, but that is not the primary validated path for this repository.

## Dataset

Training validation used a LeRobot v3 dataset.

For offline local training, point `dataset.root` to the dataset repo directory itself:

```bash
--dataset.repo_id=javadcc/evorl_screw_147 \
--dataset.root=/path/to/lerobot_cache/javadcc/evorl_screw_147 \
--dataset.revision=v3.0
```

Do not point `dataset.root` only to the global cache base directory on restricted machines.

## Checkpoint Format

Weights are saved in the LeRobot policy format:

- `config.json`
- `model.safetensors`
- `policy_preprocessor.json`
- `policy_postprocessor.json`

For stage handoff, `--policy.path` must point to:

- `.../checkpoints/<step>/pretrained_model`

## Stage Handoff

Use:

- `--policy.path=<stage1_checkpoint_dir>/pretrained_model`
- `--resume_pretrain=true`

Meaning:

- load model weights
- do not restore optimizer state
- do not restore scheduler state
- do not restore global step

## Reference Training Commands

These commands assume:

- repository root is the current directory
- `PYTHONPATH=.`
- InternVL3 weights are already available
- `torchcodec` runtime is configured

The parameter values below are aligned with the current Evo1 LeRobot setup:

- `dropout=0.2`
- `optimizer_lr=1e-5`
- `optimizer_weight_decay=1e-3`
- `scheduler_warmup_steps=1000`
- `num_layers=8`
- `chunk_size=50`
- `n_action_steps=50`
- `max_action_dim=24`
- `max_state_dim=24`

`num_layers=8` is the number of transformer blocks in the flow-matching action head. It is not the number of InternVL layers.

### Stage 1

```bash
cd /path/to/Evo-1
PYTHONPATH=evo1_lerobot accelerate launch \
  --config_file evo1_lerobot/lerobot/policies/evo1/accelerate_four_gpu_bf16.yaml \
  -m lerobot.scripts.lerobot_train \
  --dataset.repo_id=<your_dataset_repo> \
  --dataset.root=<your_dataset_repo_root> \
  --dataset.revision=v3.0 \
  --dataset.video_backend=torchcodec \
  --policy.type=evo1 \
  --policy.training_stage=stage1 \
  --policy.vlm_model_name=<path_or_hf_id_to_InternVL3-1B> \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --policy.dropout=0.2 \
  --policy.optimizer_lr=1e-5 \
  --policy.optimizer_weight_decay=1e-3 \
  --policy.optimizer_grad_clip_norm=1.0 \
  --policy.scheduler_warmup_steps=1000 \
  --policy.num_layers=8 \
  --policy.chunk_size=50 \
  --policy.n_action_steps=50 \
  --policy.max_action_dim=24 \
  --policy.max_state_dim=24 \
  --batch_size=16 \
  --num_workers=16 \
  --steps=5000 \
  --log_freq=10 \
  --save_checkpoint=true \
  --save_freq=2500 \
  --eval_freq=0 \
  --wandb.enable=false \
  --output_dir=<output_dir_stage1>
```

### Stage 2

```bash
cd /path/to/Evo-1
PYTHONPATH=evo1_lerobot accelerate launch \
  --config_file evo1_lerobot/lerobot/policies/evo1/accelerate_four_gpu_bf16.yaml \
  -m lerobot.scripts.lerobot_train \
  --dataset.repo_id=<your_dataset_repo> \
  --dataset.root=<your_dataset_repo_root> \
  --dataset.revision=v3.0 \
  --dataset.video_backend=torchcodec \
  --policy.path=<output_dir_stage1>/checkpoints/005000/pretrained_model \
  --policy.training_stage=stage2 \
  --resume_pretrain=true \
  --policy.vlm_model_name=<path_or_hf_id_to_InternVL3-1B> \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --policy.dropout=0.2 \
  --policy.optimizer_lr=1e-5 \
  --policy.optimizer_weight_decay=1e-3 \
  --policy.optimizer_grad_clip_norm=1.0 \
  --policy.scheduler_warmup_steps=1000 \
  --policy.num_layers=8 \
  --policy.chunk_size=50 \
  --policy.n_action_steps=50 \
  --policy.max_action_dim=24 \
  --policy.max_state_dim=24 \
  --batch_size=16 \
  --num_workers=16 \
  --steps=10000 \
  --log_freq=10 \
  --save_checkpoint=true \
  --save_freq=10000 \
  --eval_freq=0 \
  --wandb.enable=false \
  --output_dir=<output_dir_stage2>
```
