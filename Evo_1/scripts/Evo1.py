import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from typing import List, Union, Tuple
from PIL import Image
import torch
import torch.nn as nn
from model.internvl3.internvl3_embedder import InternVL3Embedder
from model.action_head.flow_matching import FlowmatchingActionHead
from config import EvoConfig
import logging

class EVO1(nn.Module):
    def __init__(self, config: EvoConfig):
        super().__init__() 
        self.config = config
        self._device = config.device
        self.return_cls_only = config.return_cls_only
        vlm_name = config.vlm_name
        self.embedder = InternVL3Embedder(
            model_name=vlm_name,
            device=self._device,
            enable_gradient_checkpointing=config.enable_gradient_checkpointing,
            gradient_checkpointing_use_reentrant=config.gradient_checkpointing_use_reentrant,
        )

        action_head_type = config.action_head.lower()
        
        if action_head_type == "flowmatching":
           
            horizon = config.action_horizon if config.action_horizon is not None else config.horizon
            per_action_dim = config.per_action_dim
            action_dim = config.action_dim
            
            if action_dim != horizon * per_action_dim:
                raise ValueError(f"action_dim ({action_dim}) ≠ horizon ({horizon}) × per_action_dim ({per_action_dim})")
            
            self.horizon = horizon
            self.per_action_dim = per_action_dim
            
            self.action_head = FlowmatchingActionHead(config=config).to(self._device)
        else:
            raise NotImplementedError(f"Unknown action_head: {action_head_type}")

    def get_vl_embeddings(
        self,
        images: List[List[Union[Image.Image, torch.Tensor]]],
        image_mask: torch.Tensor,  
        prompt: List[str] = None,
        return_cls_only: Union[bool, None] = None
    ) -> torch.Tensor:

        if return_cls_only is None:
            return_cls_only = self.return_cls_only

        if images is None or len(images) == 0:
            raise ValueError("Must provide at least one batch of images. Got `images=None` or empty list.")
        
        if prompt is None:
            prompt = [""] * len(images)
            
        return self.embedder.get_fused_image_text_embedding_from_tensor_images(
            image_tensors_batch=images,
            image_masks=image_mask,
            text_prompts=prompt,
            return_cls_only=return_cls_only,
        )

    def prepare_state(self, state_input: Union[list, torch.Tensor]) -> torch.Tensor:

        if isinstance(state_input, list):
            state_tensor = torch.tensor(state_input)
        elif isinstance(state_input, torch.Tensor):
            state_tensor = state_input
        else:
            raise TypeError("Unsupported state input type")

        if state_tensor.ndim == 1:
            state_tensor = state_tensor.unsqueeze(0)

        return state_tensor.to(self._device)

    
    def predict_action(
        self,
        fused_tokens: torch.Tensor,
        state: torch.Tensor,
        actions_gt: torch.Tensor = None,
        action_mask: torch.Tensor = None,
        embodiment_ids: torch.Tensor = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        
        if actions_gt is None:
            return self.action_head.get_action(fused_tokens, state=state, action_mask=action_mask, embodiment_id=embodiment_ids)
        else:
            return self.action_head(fused_tokens, state=state, actions_gt=actions_gt, action_mask=action_mask, embodiment_id=embodiment_ids)


    @torch.no_grad()
    def run_inference(
        self,
        images: List[Union[Image.Image, torch.Tensor]],
        image_mask: torch.Tensor,
        prompt: str,
        state_input: Union[list, torch.Tensor],
        return_cls_only: Union[bool, None] = None,
        action_mask: Union[torch.Tensor, None] = None
    ) -> torch.Tensor:

        # Wrap inputs into a batch of size 1 for the embedder
        if image_mask.dim() == 1:
            image_mask = image_mask.unsqueeze(0)
            
        fused_tokens = self.get_vl_embeddings(
                        images=[images],
                        image_mask=image_mask,
                        prompt=[prompt],
                        return_cls_only=return_cls_only
                        
                    )

        state_tensor = self.prepare_state(state_input)  
        
        action = self.predict_action(fused_tokens, state_tensor, action_mask=action_mask)

        if isinstance(action, torch.Tensor) and action.dtype == torch.bfloat16:
            action = action.to(torch.float32)
        
        return action
    

    def forward(self, fused_tokens, state=None, actions_gt=None, action_mask=None, embodiment_ids=None):
   

        return self.predict_action(fused_tokens, state, actions_gt, action_mask, embodiment_ids)

    def _freeze_module(self, module: nn.Module, name: str):
        print(f"Freezing {name} parameters...")
        for p in module.parameters():
            p.requires_grad = False

    def _set_module_trainable(self, module: nn.Module, trainable: bool, name: str):
        action = "Finetuning" if trainable else "Freezing"
        print(f"{action} {name} parameters...")
        for p in module.parameters():
            p.requires_grad = trainable

    def set_finetune_flags(self):
        config = self.config  
        legacy_finetune_vlm = config.finetune_vlm
        finetune_language_model = config.finetune_language_model
        finetune_vision_model = config.finetune_vision_model

        has_explicit_branch_flags = finetune_language_model or finetune_vision_model

        if has_explicit_branch_flags:
            self._set_module_trainable(self.embedder, False, "VLM (InternVL3)")
            self._set_module_trainable(self.embedder.model.language_model, finetune_language_model, "VLM language_model")
            self._set_module_trainable(self.embedder.model.vision_model, finetune_vision_model, "VLM vision_model")
            if hasattr(self.embedder.model, "mlp1"):
                self._set_module_trainable(self.embedder.model.mlp1, finetune_vision_model, "VLM visual projector (mlp1)")
        else:
            if not legacy_finetune_vlm:
                self._freeze_module(self.embedder, "VLM (InternVL3)")
            else:
                print("Finetuning VLM (InternVL3)...")

        if not config.finetune_action_head:
            self._freeze_module(self.action_head, "Action Head")
        else:
            print("Finetuning Action Head...")
