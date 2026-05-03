#!/usr/bin/env python

import argparse
import inspect
import json
from pathlib import Path

import torch

from lerobot.configs.default import DatasetConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.factory import make_dataset
from lerobot.policies.factory import make_policy, make_policy_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-repo-id", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--dataset-revision", default="v3.0")
    parser.add_argument("--video-backend", default="torchcodec")
    parser.add_argument("--vlm-model-name", required=True)
    parser.add_argument("--training-stage", choices=("stage1", "stage2"), required=True)
    parser.add_argument("--pretrained-path")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--chunk-size", type=int, default=50)
    parser.add_argument("--n-action-steps", type=int, default=50)
    parser.add_argument("--max-action-dim", type=int, default=24)
    parser.add_argument("--max-state-dim", type=int, default=24)
    parser.add_argument("--num-layers", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--enable-gradient-checkpointing", action="store_true")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-tensors")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def tensor_stats(tensor: torch.Tensor) -> dict:
    flat = tensor.detach().float().cpu().reshape(-1)
    preview = flat[:16].tolist()
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "mean": float(flat.mean().item()),
        "std": float(flat.std(unbiased=False).item()),
        "norm": float(flat.norm().item()),
        "preview": preview,
    }


def main() -> None:
    args = parse_args()

    dataset_cfg = DatasetConfig(
        repo_id=args.dataset_repo_id,
        root=args.dataset_root,
        revision=args.dataset_revision,
        video_backend=args.video_backend,
    )
    dataset_cfg.image_transforms.enable = False

    policy_kwargs = {
        "training_stage": args.training_stage,
        "vlm_model_name": args.vlm_model_name,
        "device": args.device,
        "chunk_size": args.chunk_size,
        "n_action_steps": args.n_action_steps,
        "max_action_dim": args.max_action_dim,
        "max_state_dim": args.max_state_dim,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "pretrained_path": Path(args.pretrained_path) if args.pretrained_path else None,
    }
    evo1_signature = inspect.signature(make_policy_config("evo1").__class__)
    if "enable_gradient_checkpointing" in evo1_signature.parameters:
        policy_kwargs["enable_gradient_checkpointing"] = args.enable_gradient_checkpointing
    policy_cfg = make_policy_config("evo1", **policy_kwargs)
    train_cfg = TrainPipelineConfig(
        dataset=dataset_cfg,
        policy=policy_cfg,
        output_dir=Path("/tmp/flash_port_probe_unused_output"),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        steps=1,
        eval_freq=0,
        log_freq=1,
        save_checkpoint=False,
        use_policy_training_preset=True,
        seed=args.seed,
    )

    set_seed(args.seed)
    dataset = make_dataset(train_cfg)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=getattr(policy_cfg, "drop_last", False),
    )
    batch = next(iter(dataloader))

    set_seed(args.seed)
    policy = make_policy(policy_cfg, ds_meta=dataset.meta)
    policy.to(args.device)

    prompts = policy._normalize_task_batch(batch)
    image_batches, image_masks = policy._collect_image_batches(batch)

    policy.eval()
    with torch.no_grad():
        set_seed(args.seed)
        fused_tokens = policy._compute_fused_tokens(prompts, image_batches, image_masks)

        set_seed(args.seed)
        loss, loss_info = policy.forward(batch)

        set_seed(args.seed)
        action_chunk = policy.predict_action_chunk(batch)

        set_seed(args.seed)
        policy._action_queue.clear()
        selected_action = policy.select_action(batch)

    result = {
        "training_stage": args.training_stage,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "task_preview": prompts[:2],
        "loss": float(loss.detach().float().mean().item()),
        "loss_info": loss_info,
        "fused_tokens": tensor_stats(fused_tokens),
        "action_chunk": tensor_stats(action_chunk),
        "selected_action": tensor_stats(selected_action),
    }

    if args.output_tensors:
        tensor_path = Path(args.output_tensors)
        tensor_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "fused_tokens": fused_tokens.detach().float().cpu(),
                "action_chunk": action_chunk.detach().float().cpu(),
                "selected_action": selected_action.detach().float().cpu(),
                "loss": torch.as_tensor(float(result["loss"])),
            },
            tensor_path,
        )
        result["tensor_path"] = str(tensor_path)

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
