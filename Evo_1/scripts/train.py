import sys
import os
import math
from torch import amp
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import time
import wandb
import swanlab
import torch
import torch.nn as nn
import inspect
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.optim.lr_scheduler import LambdaLR
from Evo1 import EVO1
from accelerate import Accelerator 
import logging
from datetime import datetime
import argparse
from accelerate import Accelerator, DistributedType
import json
import shutil
from torch.optim import AdamW

from config import EvoConfig
import warnings

accelerator = Accelerator()

def inspect_named_submodules(module_dict: dict, verbose: bool = True):

    total_all, trainable_all = 0, 0
    logging.info("\n Parameter Inspection by Module:")
    logging.info("=" * 70)
    for module_name, module in module_dict.items():
        total, trainable = 0, 0
        logging.info(f"\n Module: {module_name}")
        logging.info("-" * 70)
        for name, param in module.named_parameters():
            num_params = param.numel()
            total += num_params
            if param.requires_grad:
                trainable += num_params
                if verbose:
                    logging.info(f"Trainable {name:55s} | shape: {str(tuple(param.shape)):20s} | {num_params/1e6:6.2f}M")
            elif verbose:
                logging.info(f"Frozen {name:55s} | shape: {str(tuple(param.shape)):20s} | {num_params/1e6:6.2f}M")
        logging.info("-" * 70)
        logging.info(f"Total     : {total / 1e6:.2f}M")
        logging.info(f"Trainable : {trainable / 1e6:.2f}M")
        logging.info(f"Frozen    : {(total - trainable) / 1e6:.2f}M")
        total_all += total
        trainable_all += trainable
    logging.info("=" * 70)
    logging.info(f"ALL TOTAL     : {total_all / 1e6:.2f}M")
    logging.info(f"ALL TRAINABLE : {trainable_all / 1e6:.2f}M")
    logging.info(f"ALL FROZEN    : {(total_all - trainable_all) / 1e6:.2f}M")
    logging.info("=" * 70)


def custom_collate_fn(batch):
    prompts = [item["prompt"] for item in batch]
    images = [item["images"] for item in batch]
    states = torch.stack([item["state"] for item in batch], dim=0)
    actions = torch.stack([item["action"] for item in batch], dim=0)
    action_mask = torch.stack([item["action_mask"] for item in batch], dim=0)
    image_masks = torch.stack([item["image_mask"] for item in batch], dim=0)
    state_mask = torch.stack([item["state_mask"] for item in batch], dim=0)
    embodiment_ids = torch.stack([item["embodiment_id"] for item in batch], dim=0)

    return {
        "prompts": prompts,
        "images": images,
        "states": states,
        "actions": actions,
        "action_mask": action_mask,
        "state_mask": state_mask,
        "image_masks": image_masks,
        "embodiment_ids": embodiment_ids
    }

def get_lr_lambda(warmup_steps, total_steps, resume_step=0):
    def lr_lambda(current_step):
        current_step += resume_step  
        if current_step < warmup_steps:
            return current_step / max(1, warmup_steps)
        progress = (current_step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return lr_lambda
    
def setup_logging(log_dir: str) -> str:
    from datetime import datetime
    import logging, os

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"train_log_{timestamp}.log")
    if accelerator is None or accelerator.is_main_process:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[
                logging.FileHandler(log_path),
                logging.StreamHandler()
            ]
        )
        logging.info(f"Logging to: {log_path}")
    return log_path

def init_wandb(config: EvoConfig, accelerator: Accelerator):

    if accelerator.is_main_process:
        if config.disable_wandb:
            os.environ["WANDB_MODE"] = "disabled"

        wandb.init(
            project=config.wandb_project,
            name=config.run_name,
            config=config.to_dict(),
            dir=config.save_dir,
            mode="offline",
        )

        wandb.define_metric("step")
        wandb.define_metric("*", step_metric="step")

def init_swanlab(config: EvoConfig, accelerator: Accelerator):

    if accelerator is None or accelerator.is_main_process:
        swanlab.init(
            project=config.wandb_project,
            name=config.run_name,
            config=config.to_dict()
        )

def prepare_dataset(config: EvoConfig) -> torch.utils.data.Dataset:
    dataset_type = config.dataset_type
    image_size = config.image_size
    max_samples = config.max_samples_per_file
    horizon = config.horizon
    binarize_gripper = config.binarize_gripper
    use_augmentation = config.use_augmentation
    
    if dataset_type == "lerobot":
        from dataset.lerobot_dataset_pretrain_mp import LeRobotDataset 
        import yaml
        with open(config.dataset_config_path, 'r') as f:
            dataset_config = yaml.safe_load(f)

        dataset = LeRobotDataset(
            config=dataset_config,
            image_size=image_size,
            max_samples_per_file=max_samples,
            action_horizon=horizon,
            binarize_gripper=binarize_gripper,
            use_augmentation=use_augmentation,
            video_backend=config.video_backend,
            cache_dir=config.cache_dir if config.cache_dir else os.path.join(os.path.dirname(__file__), "..", "dataset", "dataset_cache")
        )
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}")
    if accelerator is None or accelerator.is_main_process:
        logging.info(f"Loaded {len(dataset)} samples from {config.data_paths} ({dataset_type})")
    return dataset


def prepare_dataloader(dataset, config: EvoConfig) -> DataLoader:
    batch_size = config.batch_size
    num_workers = config.num_workers
    pin_memory = config.pin_memory
    persistent_workers = config.persistent_workers
    prefetch_factor = config.prefetch_factor

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        drop_last=True,
        collate_fn=custom_collate_fn
    )
    if accelerator is None or accelerator.is_main_process:
        logging.info(f"Initialized dataloader with batch size {batch_size}, num_workers {num_workers}, pin_memory {pin_memory}, persistent_workers {persistent_workers}, prefetch_factor {prefetch_factor}")
    return dataloader


def check_numerical_stability(step: int, **named_tensors) -> bool:
    for name, tensor in named_tensors.items():
        if tensor is not None and not torch.isfinite(tensor).all():
            if accelerator.is_main_process:
                logging.info(f"[Step {step}] Non-finite detected in {name}")
            return False
    return True

def dump_numerical_issue(save_dir: str, step: int, **named_tensors) -> str:
    os.makedirs(save_dir, exist_ok=True)
    payload = {
        "step": int(step),
        "captured_at": datetime.now().isoformat(),
        "tensors": {},
    }
    for name, tensor in named_tensors.items():
        if isinstance(tensor, torch.Tensor):
            detached = tensor.detach()
            payload["tensors"][name] = {
                "shape": list(detached.shape),
                "dtype": str(detached.dtype),
                "device": str(detached.device),
                "is_finite": bool(torch.isfinite(detached).all().item()),
                "min": float(detached.min().item()) if detached.numel() > 0 else 0.0,
                "max": float(detached.max().item()) if detached.numel() > 0 else 0.0,
                "mean": float(detached.float().mean().item()) if detached.numel() > 0 else 0.0,
            }
    out_path = os.path.join(save_dir, f"numerical_issue_step_{step}.json")
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return out_path

def log_training_step(step, loss, total_norm, momentum_norm, scheduler, dataloader, accelerator, sec_per_step=None, samples_per_sec=None, it_per_sec=None, window_max_allocated_gb=None, window_max_reserved_gb=None, global_max_allocated_gb=None, global_max_reserved_gb=None):
    current_epoch = step / len(dataloader)
    
    # === Collect GPU memory stats ===
    current_real_memory = torch.cuda.memory_allocated() / 1024**3
    peak_real_memory = torch.cuda.max_memory_allocated() / 1024**3
    current_reserved_memory = torch.cuda.memory_reserved() / 1024**3
    
    if accelerator is None or accelerator.is_main_process:
        logging.info(f"Estimated Epoch: {current_epoch:.2f}")
        logging.info(f"[Step {step}] Loss: {loss.item():.4f}")
        if sec_per_step is not None:
            logging.info(f"Speed: {sec_per_step:.4f} s/it | {it_per_sec:.2f} it/s | {samples_per_sec:.2f} samples/s")
        if window_max_allocated_gb is not None:
            logging.info(f"Memory window peak: allocated={window_max_allocated_gb:.2f} GiB reserved={window_max_reserved_gb:.2f} GiB")
        if global_max_allocated_gb is not None:
            logging.info(f"Memory global peak: allocated={global_max_allocated_gb:.2f} GiB reserved={global_max_reserved_gb:.2f} GiB")
        logging.info(f"[Step {step}] GPU Memory - Current: {current_real_memory:.2f} GB, Peak: {peak_real_memory:.2f} GB, Reserved: {current_reserved_memory:.2f} GB")
        
        wandb_log_dict = {
            "step": step,
            "loss": loss.item(),
            "current_epoch": current_epoch,
            "learning_rate": scheduler.get_last_lr()[0],
            "grad_norm/total": total_norm,
            "optimizer/momentum_norm": momentum_norm.item(),
            "memory/current_gb": current_real_memory,
            "memory/peak_gb": peak_real_memory,
            "memory/reserved_gb": current_reserved_memory,
        }
        if sec_per_step is not None:
            wandb_log_dict["sec_per_step"] = sec_per_step
            wandb_log_dict["it_per_sec"] = it_per_sec
            wandb_log_dict["samples_per_sec"] = samples_per_sec
        if window_max_allocated_gb is not None:
            wandb_log_dict["window_max_allocated_gb"] = window_max_allocated_gb
            wandb_log_dict["window_max_reserved_gb"] = window_max_reserved_gb
        if global_max_allocated_gb is not None:
            wandb_log_dict["global_max_allocated_gb"] = global_max_allocated_gb
            wandb_log_dict["global_max_reserved_gb"] = global_max_reserved_gb

        wandb.log(wandb_log_dict)
        swanlab.log(wandb_log_dict)

def save_checkpoint(save_dir, step, model_engine, loss, accelerator, optimizer=None, scheduler=None, config: EvoConfig=None, norm_stats=None):
    tag = f"step_{step}"
    checkpoint_dir = os.path.join(save_dir, tag)
    use_deepspeed = accelerator.distributed_type == DistributedType.DEEPSPEED

    if accelerator.is_main_process and os.path.exists(checkpoint_dir):
        logging.warning(f"Checkpoint directory {checkpoint_dir} exists. Removing before overwrite.")
        shutil.rmtree(checkpoint_dir)

    accelerator.wait_for_everyone()

    client_state = {
        "step": step,
        "best_loss": loss if isinstance(loss, float) else loss.item(),
        "config": config.to_dict() if config else None,
    } if accelerator.is_main_process else {} 

    if use_deepspeed:
        model_engine.save_checkpoint(save_dir, tag=tag, client_state=client_state)
        if accelerator.is_main_process:
            checkpoint_meta = {
                "type": "ds_model",
                "version": 0.0,
                "checkpoints": "mp_rank_00_model_states.pt"
            }
            with open(os.path.join(checkpoint_dir, "checkpoint.json"), "w") as f:
                json.dump(checkpoint_meta, f, indent=2)
    else:
        if accelerator.is_main_process:
            os.makedirs(checkpoint_dir, exist_ok=True)
            unwrapped_model = accelerator.unwrap_model(model_engine)
            payload = {
                "step": step,
                "best_loss": loss if isinstance(loss, float) else loss.item(),
                "config": config.to_dict() if config else None,
                "norm_stats": norm_stats,
                "model_state_dict": unwrapped_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
                "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            }
            torch.save(payload, os.path.join(checkpoint_dir, "checkpoint.pt"))
            checkpoint_meta = {
                "type": "torch_model",
                "version": 1.0,
                "checkpoint_file": "checkpoint.pt",
            }
            with open(os.path.join(checkpoint_dir, "checkpoint.json"), "w") as f:
                json.dump(checkpoint_meta, f, indent=2)

    if accelerator.is_main_process:
        if config is not None:
            config_path = os.path.join(checkpoint_dir, "config.json")
            with open(config_path, "w") as f:
                json.dump(config.to_dict(), f, indent=2)

        if norm_stats is not None:
            norm_stats_path = os.path.join(checkpoint_dir, "norm_stats.json")
            with open(norm_stats_path, "w") as f:
                json.dump(norm_stats, f, indent=2)
                
        logging.info(f"[Rank {accelerator.process_index}] Saved checkpoint to {checkpoint_dir}")

def load_checkpoint_standard(model_engine, optimizer, load_dir, accelerator, tag="step_best", load_optimizer_states=True):
    checkpoint_path = os.path.join(load_dir, tag, "checkpoint.pt")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Standard checkpoint file not found: {checkpoint_path}")

    payload = torch.load(checkpoint_path, map_location="cpu")
    unwrapped_model = accelerator.unwrap_model(model_engine)
    
    try:
        unwrapped_model.load_state_dict(payload["model_state_dict"], strict=True)
    except Exception as e:
        if accelerator.is_main_process:
            logging.warning(f"Strict load failed: {e}. Trying non-strict.")
        unwrapped_model.load_state_dict(payload["model_state_dict"], strict=False)

    if load_optimizer_states and optimizer is not None and payload.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(payload["optimizer_state_dict"])

    if accelerator.is_main_process:
        logging.info(f"Loaded standard checkpoint from {checkpoint_path}")
    return payload.get("step", 0), payload

def load_checkpoint_with_deepspeed(model_engine, load_dir, accelerator, tag="step_best", load_optimizer_states=True, resume_pretrain=False):

    try:
        load_path, client_state = model_engine.load_checkpoint(
            load_dir,
            tag=tag,
            load_module_strict=True,
            load_optimizer_states=load_optimizer_states and not resume_pretrain,
            load_lr_scheduler_states=load_optimizer_states and not resume_pretrain
        )
        if accelerator.is_main_process:
            logging.info(f"Loaded DeepSpeed checkpoint from {load_dir}/{tag} (including optimizer states)")
        return client_state.get("step", 0), client_state
        
    except Exception as e:
        if accelerator.is_main_process:
            logging.warning(f"World size mismatch detected: {str(e)}")
            logging.warning("Attempting to load only model weights (skipping optimizer states)...")
        try:
            load_path, client_state = model_engine.load_checkpoint(
                load_dir,
                tag=tag,
                load_module_strict=True,
                load_optimizer_states=False,
                load_lr_scheduler_states=False
            )
            if accelerator.is_main_process:
                logging.info(f"Loaded DeepSpeed checkpoint from {load_dir}/{tag} (model weights only)")
            return client_state.get("step", 0), client_state
            
        except Exception as e2:
            if accelerator.is_main_process:
                logging.error(f"Failed to load checkpoint even without optimizer states: {str(e2)}")
            raise RuntimeError(f"Failed to load DeepSpeed checkpoint from {load_dir} with tag {tag}: {str(e2)}")

    


# def get_and_clip_grad_norm(accelerator, model, max_norm: float = 1.0):
#     """
#     Clips gradient norm and returns the total norm before clipping.
#     """
#     total_norm = accelerator.clip_grad_norm_(model.parameters(), max_norm)
#     return total_norm

def get_and_clip_grad_norm(accelerator, model, loss, max_norm: float = 1.0):

    if hasattr(accelerator, "get_global_grad_norm") and hasattr(accelerator, "clip_grad_norm_"):
       
        total_norm = accelerator.get_global_grad_norm()
        accelerator.clip_grad_norm_(model.parameters(), max_norm)
        clipped_norm = accelerator.get_global_grad_norm()
    else:
 
        grad_norms = [p.grad.norm(2) for p in model.parameters() if p.grad is not None]
        if len(grad_norms) == 0:
            total_norm = torch.tensor(0.0, device=loss.device)
        else:
            total_norm = torch.norm(torch.stack(grad_norms), 2)

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

        clipped_grad_norms = [p.grad.norm(2) for p in model.parameters() if p.grad is not None]
        if len(clipped_grad_norms) == 0:
            clipped_norm = torch.tensor(0.0, device=loss.device)
        else:
            clipped_norm = torch.norm(torch.stack(clipped_grad_norms), 2)

    return total_norm, clipped_norm

def get_optimizer_momentum_norm(optimizer: torch.optim.Optimizer, accelerator) -> torch.Tensor:
    """
    compute L2 norm of optimizer momentum buffers (first order exp_avg for AdamW)
    Works with accelerate-wrapped optimizers.
    """
    if hasattr(optimizer, 'optimizer'):
        base_optimizer = optimizer.optimizer
        print("Detected AcceleratedOptimizer wrapper.")
    else:
        base_optimizer = optimizer
    
    momentum_norms = []
    for group in base_optimizer.param_groups:
        for p in group['params']:
            state = base_optimizer.state.get(p)
            if state is not None and 'exp_avg' in state:
                momentum_buffer = state['exp_avg']
                momentum_norms.append(momentum_buffer.norm(2).item())

    if not momentum_norms:
        return torch.tensor(0.0)

    total_momentum_norm = torch.tensor(momentum_norms).norm(2)
    return total_momentum_norm

def build_param_groups(model, wd):
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad: 
            continue
        is_bias = n.endswith("bias") or ".bias" in n
        is_norm = (p.dim() == 1) or ("norm" in n.lower())
        (no_decay if is_bias or is_norm else decay).append(p)
    return [{"params": decay, "weight_decay": wd},
            {"params": no_decay, "weight_decay": 0.0}]

def train(config: EvoConfig):


    # === Set logging ===
    save_dir = config.save_dir
    log_path = setup_logging(save_dir)
    
    # === WandB and Swanlab ===
    init_wandb(config, accelerator)
    init_swanlab(config, accelerator)

    # === Debug mode ===
    if config.debug:
        torch.autograd.set_detect_anomaly(True)

    # === Dataset ===
    dataset = prepare_dataset(config)

    # === DataLoader ===
    dataloader = prepare_dataloader(dataset, config)

    # === Model ===
    model = EVO1(config)
    model.train()
    model.set_finetune_flags()

    lr = config.lr
    wd = config.weight_decay
    
    use_fused_adamw = config.fused_adamw
    optimizer_kwargs = {}
    if use_fused_adamw and torch.cuda.is_available():
        if "fused" in inspect.signature(torch.optim.AdamW).parameters:
            optimizer_kwargs["fused"] = True
            if accelerator.is_main_process:
                logging.info("Enabled fused AdamW optimizer")
        else:
            if accelerator.is_main_process:
                logging.warning("Fused AdamW requested but not supported by this PyTorch version")

    optimizer = AdamW(build_param_groups(model, wd), lr=lr, **optimizer_kwargs)
    if accelerator.is_main_process:
        logging.info(f"Optimizer=AdamW, lr={lr}, weight_decay={wd}, fused={optimizer_kwargs.get('fused', False)}")


    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
    model_engine = model  
  
    if accelerator.is_main_process:
        logging.info("Initialized with Accelerate")
    
    
    # === Warmup + Cosine Scheduler ===
    max_steps = config.max_steps
    warmup_steps = config.warmup_steps
    
    # === loss function ===
    loss_fn = nn.MSELoss() 

    # === Checkpoint and save path setup ===
    os.makedirs(save_dir, exist_ok=True)
    best_ckpt_path = os.path.join(save_dir, "best_checkpoint.pt")
    best_loss = float("inf")
    
    # === Logging and interval settings ===
    log_interval = config.log_interval
    ckpt_interval = config.ckpt_interval
    max_norm = config.grad_clip_norm

    # === Resume training from checkpoint ===
    resume = config.resume
    resume_path = config.resume_path
    resume_pretrain = config.resume_pretrain

    if resume != bool(resume_path):
        raise ValueError("Inconsistent resume configuration: --resume and --resume_path must be set together.")
    
    if resume:
        resume_path = resume_path.rstrip("/")
        resume_dir, resume_tag = os.path.split(resume_path)
        
        if accelerator.distributed_type == DistributedType.DEEPSPEED:
            step, client_state = load_checkpoint_with_deepspeed(
                model_engine,
                load_dir=resume_dir,
                accelerator=accelerator,
                tag=resume_tag,
                load_optimizer_states=True,  
                resume_pretrain=resume_pretrain
            )
        else:
            step, client_state = load_checkpoint_standard(
                model_engine,
                optimizer=optimizer,
                load_dir=resume_dir,
                accelerator=accelerator,
                tag=resume_tag,
                load_optimizer_states=not resume_pretrain
            )
            
        best_loss = client_state.get("best_loss", float("inf"))
        if accelerator.is_main_process:
            logging.info(f"Resuming from {resume_dir}/{resume_tag}, step {step}")
    else:
        step = 0
        if accelerator.is_main_process:
            logging.info("Starting fresh training")

    if resume_pretrain:
        step = 0
        logging.info("Resuming pretraining from scratch, resetting step to 0")

    scheduler = LambdaLR(optimizer, get_lr_lambda(warmup_steps, max_steps, resume_step=step))


    if accelerator.is_main_process:
        
        inspect_named_submodules({
            "vision_model": model.embedder.model.vision_model,
            "language_model": model.embedder.model.language_model,
            "action_head": model.action_head
        })

    vision_masked = config.vision_masked
    if vision_masked and accelerator.is_main_process:
        logging.info("Vision masking is enabled: all image masks will be set to 0 during training.")

    non_finite_streak = 0
    last_log_time = time.perf_counter()
    last_log_step = step
    global_peak_allocated_gb = None
    global_peak_reserved_gb = None
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        global_peak_allocated_gb = 0.0
        global_peak_reserved_gb = 0.0

    # === Training Loop ===
    while step < max_steps:
        for batch in tqdm(dataloader, desc="Training", disable=not accelerator.is_main_process):
            torch.cuda.reset_peak_memory_stats()
            
            if step >= max_steps:
                break
            prompts = batch["prompts"]
            images_batch = batch["images"]
            image_masks = batch["image_masks"]
            if vision_masked:
                image_masks = torch.zeros_like(image_masks)
            states = batch["states"].to(dtype=torch.float32)
            actions_gt = batch["actions"].to(dtype=torch.float32)
            action_mask = batch["action_mask"]
            state_mask = batch["state_mask"]
            embodiment_ids = batch["embodiment_ids"]
            
            with accelerator.autocast():
                fused_tokens = model.get_vl_embeddings(
                    images=images_batch, 
                    image_mask=image_masks, 
                    prompt=prompts, 
                    return_cls_only=False
                )

                pred_velocity, noise = model(fused_tokens, state=states, actions_gt=actions_gt, action_mask=action_mask)
                
            target_velocity = (actions_gt - noise).view(actions_gt.shape[0], -1)
            
            assert pred_velocity.shape == target_velocity.shape

            if action_mask.sum() == 0:
                raise ValueError(f"[Step {step}] action_mask.sum() is 0! All actions are masked. "
                            f"This indicates a problem with the data or mask generation. "
                            f"action_mask shape: {action_mask.shape}, "
                            f"action_mask: {action_mask}")
            

            action_mask = action_mask.view(action_mask.shape[0], -1).to(dtype=pred_velocity.dtype)
            pred_velocity_mask = pred_velocity * action_mask
            target_velocity_mask = target_velocity * action_mask
            loss = loss_fn(pred_velocity_mask.float(), target_velocity_mask.float())
            scale_factor = action_mask.numel() / (action_mask.sum() + 1e-8)
            loss = loss * scale_factor
            
            # === NaN/Inf check ===
            is_stable = check_numerical_stability(
                step,
                states=states,
                actions_gt=actions_gt,
                fused_tokens=fused_tokens,
                pred_velocity=pred_velocity,
                loss=loss
            )
            is_stable_tensor = torch.tensor(1 if is_stable else 0, device=accelerator.device, dtype=torch.long)
            if accelerator.distributed_type != DistributedType.NO:
                torch.distributed.all_reduce(is_stable_tensor, op=torch.distributed.ReduceOp.MIN)
            
            if is_stable_tensor.item() == 0:
                out_path = dump_numerical_issue(
                    save_dir=os.path.join(save_dir, "numerical_issues"),
                    step=step,
                    states=states,
                    actions_gt=actions_gt,
                    fused_tokens=fused_tokens,
                    pred_velocity=pred_velocity,
                    loss=loss
                )
                if accelerator.is_main_process:
                    logging.warning(f"[Step {step}] Numerical instability detected across GPUs (synced). Data dumped to {out_path}.")
                
                non_finite_streak += 1
                if non_finite_streak > getattr(config, "non_finite_max_streak", 5):
                    if accelerator.is_main_process:
                        logging.error(f"[Step {step}] Numerical instability persisted for {non_finite_streak} steps. Terminating training.")
                    break
                continue
            
            non_finite_streak = 0

            # === Backward and optimizer step ===
            optimizer.zero_grad(set_to_none=True)
            accelerator.backward(loss)

            # === Clip grad norm ===
            total_norm, clipped_norm = get_and_clip_grad_norm(accelerator, model, loss, max_norm)
            # total_norm = get_and_clip_grad_norm(accelerator, model, max_norm)

            optimizer.step()
            scheduler.step()
            
            # === Logging ===
            if step % log_interval == 0:
                now = time.perf_counter()
                elapsed = now - last_log_time
                window = max(1, step - last_log_step)
                sec_per_step = elapsed / window
                
                # compute samples/sec assuming 1 backward per iteration
                batch_sz = actions_gt.shape[0] * accelerator.num_processes
                samples_per_sec = batch_sz / max(sec_per_step, 1e-8)
                it_per_sec = 1.0 / max(sec_per_step, 1e-8)
                
                window_max_allocated_gb = None
                window_max_reserved_gb = None
                if torch.cuda.is_available():
                    window_max_allocated_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
                    window_max_reserved_gb = torch.cuda.max_memory_reserved() / (1024 ** 3)
                    global_peak_allocated_gb = max(global_peak_allocated_gb, window_max_allocated_gb) if global_peak_allocated_gb is not None else window_max_allocated_gb
                    global_peak_reserved_gb = max(global_peak_reserved_gb, window_max_reserved_gb) if global_peak_reserved_gb is not None else window_max_reserved_gb
                
                momentum_norm = get_optimizer_momentum_norm(optimizer, accelerator)
                log_training_step(
                    step, loss, total_norm, momentum_norm, scheduler, dataloader, accelerator,
                    sec_per_step=sec_per_step,
                    samples_per_sec=samples_per_sec,
                    it_per_sec=it_per_sec,
                    window_max_allocated_gb=window_max_allocated_gb,
                    window_max_reserved_gb=window_max_reserved_gb,
                    global_max_allocated_gb=global_peak_allocated_gb,
                    global_max_reserved_gb=global_peak_reserved_gb
                )
                
                last_log_time = now
                last_log_step = step
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
   
            # === Save best checkpoint ===
            loss_value = loss.item()
            if accelerator.is_main_process:
                is_best = loss_value < best_loss
                if is_best:
                    best_loss = loss_value
                is_best_tensor = torch.tensor(int(is_best), device=accelerator.device)
            else:
                is_best_tensor = torch.tensor(0, device=accelerator.device)
            
            if accelerator.distributed_type != DistributedType.NO:
                torch.distributed.broadcast(is_best_tensor, src=0)
            
            if is_best_tensor.item() == 1 and step > 1000:
                accelerator.print("start to save best checkpoint")
                save_checkpoint(
                    save_dir,
                    step="best",
                    model_engine=model_engine,
                    loss=loss,
                    accelerator=accelerator,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    config=config,
                    norm_stats=dataset.arm2stats_dict 
                )
                accelerator.print("end to save best checkpoint")
                if accelerator.is_main_process:
                    logging.info(f"Saved best checkpoint at step {step} with loss {loss_value:.6f}")

            step += 1

            # === Save periodic checkpoint ===
            if step % ckpt_interval == 0 and step > 0:
                checkpoint_path = os.path.join(save_dir, f"checkpoint_step_{step}.pt")
                save_checkpoint(save_dir, step=step, model_engine=model_engine, loss=loss, accelerator=accelerator, optimizer=optimizer, scheduler=scheduler, config=config, norm_stats=dataset.arm2stats_dict)
         
    # === Save final model ===
    save_checkpoint(save_dir, step="final", model_engine=model_engine, loss=loss, accelerator=accelerator, optimizer=optimizer, scheduler=scheduler, config=config, norm_stats=dataset.arm2stats_dict)
    logging.info(f"Final model saved to step_final/")
    logging.info(f"Best checkpoint saved to step_best/ with loss {best_loss:.6f}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train Evo-1")

    # Basic config
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--run_name", type=str, default="default_run")
    parser.add_argument("--vlm_name", type=str, default="OpenGVLab/InternVL3-1B")
    parser.add_argument("--action_head", type=str, default="flowmatching", choices=["flowmatching"])
    parser.add_argument("--return_cls_only", action="store_true")
    parser.add_argument("--disable_wandb", action="store_true", help="Disable wandb logging.")
    parser.add_argument("--wandb_project", type=str, default="default_run", help="Project name for WandB and SwanLab")

    # Dataset
    parser.add_argument("--dataset_type", type=str, default="lerobot")
    parser.add_argument("--data_paths", type=str, required=False)
    parser.add_argument("--dataset_config_path", type=str, required=True)
    parser.add_argument("--image_size", type=int, default=448)
    parser.add_argument("--binarize_gripper", action="store_true", default=False, help="Whether to binarize gripper state/action (default: False).")
    parser.add_argument("--use_augmentation", action="store_true", help="Enable data augmentation on images")
    parser.add_argument("--vision_masked", action="store_true", help="Mask out all visual inputs during training by forcing image masks to 0")
    parser.add_argument("--video_backend", type=str, default="av", help="Video backend for decord (e.g. 'av', 'pyav)")
    parser.add_argument("--cache_dir", type=str, default=None, help="Optional cache directory for dataset manifests and preprocessed data")

    # Training
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_steps", type=int, default=600)
    parser.add_argument("--warmup_steps", type=int, default=300)
    parser.add_argument("--grad_clip_norm", type=float, default=1.0)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--disable_gradient_checkpointing", action="store_true", help="Disable gradient checkpointing for InternVL3 vision/language branches")
    parser.add_argument("--gradient_checkpointing_use_reentrant", action="store_true", help="Use reentrant mode when enabling gradient checkpointing")


    # Logging & checkpointing
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--ckpt_interval", type=int, default=10)
    parser.add_argument("--save_dir", type=str, default="./checkpoints")

    # Resume
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume_path", type=str, default=None)
    parser.add_argument("--resume_pretrain", action="store_true")
   

    # Finetuning
    parser.add_argument("--finetune_vlm", action="store_true")
    parser.add_argument("--finetune_language_model", action="store_true", help="Selectively finetune VLM language branch")
    parser.add_argument("--finetune_vision_model", action="store_true", help="Selectively finetune VLM vision branch and visual projector (mlp1)")
    parser.add_argument("--finetune_action_head", action="store_true")

    # Misc
    parser.add_argument("--per_action_dim", type=int, default=7)
    parser.add_argument("--state_dim", type=int, default=7)
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--num_layers", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--prefetch_factor", type=int, default=4, help="Prefetch factor for dataloader")
    parser.add_argument("--disable_pin_memory", action="store_true", help="Disable pin_memory in dataloader")
    parser.add_argument("--disable_persistent_workers", action="store_true", help="Disable persistent_workers in dataloader")
    parser.add_argument("--disable_fused_adamw", action="store_true", help="Disable fused AdamW optimizer")
    parser.add_argument("--non_finite_max_streak", type=int, default=5, help="Maximum number of consecutive steps with numerically unstable values before terminating")
    # dropout
    parser.add_argument("--dropout", type=float, default=0.0)

    # Performance
    parser.add_argument("--disable_tf32", action="store_true", help="Disable TF32 (default is enabled)")
    parser.add_argument("--disable_cudnn_benchmark", action="store_true", help="Disable cuDNN benchmark (default is enabled)")

    args = parser.parse_args()
    config_dict = vars(args)
    # Post-process inverse boolean flags for config
    config_dict["pin_memory"] = not config_dict.pop("disable_pin_memory", False)
    config_dict["persistent_workers"] = not config_dict.pop("disable_persistent_workers", False)
    config_dict["fused_adamw"] = not config_dict.pop("disable_fused_adamw", False)
    config_dict["enable_gradient_checkpointing"] = not config_dict.pop("disable_gradient_checkpointing", False)
    
    allow_tf32 = not config_dict.pop("disable_tf32", False)
    cudnn_benchmark = not config_dict.pop("disable_cudnn_benchmark", False)

    torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = allow_tf32
        torch.backends.cudnn.benchmark = cudnn_benchmark
    torch.set_float32_matmul_precision("high" if allow_tf32 else "highest")

    config = EvoConfig.from_dict(config_dict)

    try:
        train(config)
    except KeyboardInterrupt:
        if accelerator.is_main_process:
            logging.info("KeyboardInterrupt received. Cleaning up...")
        sys.exit(0)

