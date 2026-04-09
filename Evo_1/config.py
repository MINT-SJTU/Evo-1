import dataclasses
from dataclasses import dataclass, field, asdict
from typing import Optional

@dataclass
class EvoConfig:
    # Basic config
    device: str = "cuda"
    run_name: str = "default_run"
    vlm_name: str = "OpenGVLab/InternVL3-1B"
    action_head: str = "flowmatching"
    return_cls_only: bool = False
    disable_wandb: bool = False
    wandb_project: str = "default_run"
    debug: bool = False

    # Dataset
    dataset_type: str = "lerobot"
    data_paths: Optional[str] = None
    dataset_config_path: str = ""
    image_size: int = 448
    binarize_gripper: bool = False
    use_augmentation: bool = False
    vision_masked: bool = False
    max_samples_per_file: Optional[int] = None
    horizon: int = 16
    video_backend: str = "av"
    cache_dir: Optional[str] = None

    # Training
    lr: float = 1e-5
    batch_size: int = 8
    num_workers: int = 8
    prefetch_factor: Optional[int] = 4
    pin_memory: bool = True
    persistent_workers: bool = True
    max_steps: int = 600
    warmup_steps: int = 300
    grad_clip_norm: float = 1.0
    weight_decay: float = 1e-5
    fused_adamw: bool = True
    non_finite_max_streak: int = 5
    log_interval: int = 10
    ckpt_interval: int = 10
    save_dir: str = "./checkpoints"
    enable_gradient_checkpointing: bool = True
    gradient_checkpointing_use_reentrant: bool = False

    # Resume
    resume: bool = False
    resume_path: Optional[str] = None
    resume_pretrain: bool = False

    # Finetuning
    finetune_vlm: bool = False
    finetune_language_model: bool = False
    finetune_vision_model: bool = False
    finetune_action_head: bool = False

    # Misc / Model params
    per_action_dim: int = 7
    state_dim: int = 7
    action_horizon: Optional[int] = None
    num_layers: int = 8
    dropout: float = 0.0
    
    # Model defaults
    embed_dim: int = 896
    hidden_dim: int = 1024
    state_hidden_dim: int = 1024
    num_heads: int = 8
    num_categories: int = 1
    num_inference_timesteps: int = 50

    def to_dict(self):
        return asdict(self)
    
    @property
    def action_dim(self) -> int:
        ah = self.action_horizon if self.action_horizon is not None else self.horizon
        return ah * self.per_action_dim

    @classmethod
    def from_dict(cls, d: dict):
        valid_keys = {f.name for f in dataclasses.fields(cls)}
        filtered_d = {k: v for k, v in d.items() if k in valid_keys}
        return cls(**filtered_d)
