# model/internvl3/internvl3_embedder.py
import torch
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
from transformers import GenerationConfig
from torchvision.transforms.functional import to_pil_image
from typing import Union, List
from torch import nn
import logging
import functools

logger = logging.getLogger(__name__)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def flash_attn_is_available() -> bool:
    try:
        import flash_attn  # noqa: F401
    except ModuleNotFoundError:
        return False
    return True

# === Image Transformations ===
def build_transform(input_size):
    return T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

# === Aspect Ratio Handling ===
@functools.lru_cache(maxsize=10000)
def get_target_aspect_ratio(orig_width, orig_height, image_size, min_num, max_num):
    aspect_ratio = orig_width / orig_height
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = orig_width * orig_height
    for ratio in target_ratios:
        target_ar = ratio[0] / ratio[1]
        diff = abs(aspect_ratio - target_ar)
        if diff < best_ratio_diff:
            best_ratio_diff = diff
            best_ratio = ratio
        elif diff == best_ratio_diff and area > 0.5 * image_size**2 * ratio[0] * ratio[1]:
            best_ratio = ratio
    return best_ratio


class InternVL3Embedder(nn.Module):
    def __init__(
        self,
        model_name="OpenGVLab/InternVL3-1B",
        image_size=448,
        device="cuda",
        enable_gradient_checkpointing: bool = False,
        gradient_checkpointing_use_reentrant: bool = False,
    ):
        super().__init__()
        self.device = device
        self.image_size = image_size
        self.max_text_length = 1024  # InternVL3 supports up to 1024 tokens
        self.enable_gradient_checkpointing = bool(enable_gradient_checkpointing)
        self.gradient_checkpointing_use_reentrant = bool(gradient_checkpointing_use_reentrant)
        self.transform = build_transform(image_size)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=False)
        use_flash_attn = flash_attn_is_available()
        if not use_flash_attn:
            logger.warning("flash_attn is not installed. Falling back to standard attention.")
        self.model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            use_flash_attn=use_flash_attn,
            low_cpu_mem_usage=True,
            _fast_init=False,
        ).to(self.device) 
        
        if hasattr(self.model.language_model, 'model'):
            layers = self.model.language_model.model.layers

        else:
            layers = self.model.language_model.layers
        layers = layers[:14]

        if hasattr(self.model.language_model, 'model'):
            self.model.language_model.model.layers = torch.nn.ModuleList(layers)
        else:
            self.model.language_model.layers = torch.nn.ModuleList(layers)
        self.model.language_model.lm_head = torch.nn.Identity()
        self._configure_memory_features()

    def _configure_memory_features(self) -> None:
        checkpoint_kwargs = {"use_reentrant": self.gradient_checkpointing_use_reentrant}
        
        # silencing the PyTorch warning and enforcing the correct config.
        import torch.utils.checkpoint
        if not hasattr(torch.utils.checkpoint, "_checkpoint_patched"):
            _orig_checkpoint = torch.utils.checkpoint.checkpoint
            def _checkpoint_patched(*args, **kwargs):
                if "use_reentrant" not in kwargs:
                    kwargs["use_reentrant"] = self.gradient_checkpointing_use_reentrant
                return _orig_checkpoint(*args, **kwargs)
            torch.utils.checkpoint.checkpoint = _checkpoint_patched
            torch.utils.checkpoint._checkpoint_patched = True

        def _enable_ckpt(module) -> bool:
            if module is None or not hasattr(module, "gradient_checkpointing_enable"):
                return False
            try:
                module.gradient_checkpointing_enable(gradient_checkpointing_kwargs=checkpoint_kwargs)
            except TypeError:
                module.gradient_checkpointing_enable()
            return True

        if not self.enable_gradient_checkpointing:
            if hasattr(self.model, "vision_model") and hasattr(self.model.vision_model, "encoder"):
                self.model.vision_model.encoder.gradient_checkpointing = False
            return

        enabled_any = False
        enabled_any = _enable_ckpt(self.model) or enabled_any

        if hasattr(self.model, "vision_model") and hasattr(self.model.vision_model, "encoder"):
            self.model.vision_model.encoder.gradient_checkpointing = True
            enabled_any = True

        if hasattr(self.model, "language_model"):
            language_model = self.model.language_model
            enabled_any = _enable_ckpt(language_model) or enabled_any
            if hasattr(language_model, "model"):
                enabled_any = _enable_ckpt(language_model.model) or enabled_any
            if hasattr(language_model, "config"):
                language_model.config.use_cache = False

        if hasattr(self.model, "config"):
            self.model.config.use_cache = False

        if hasattr(self.model, "enable_input_require_grads"):
            self.model.enable_input_require_grads()

        if enabled_any:
            logger.info("Gradient checkpointing enabled for InternVL3 embedder.")
        else:
            logger.warning("Requested gradient checkpointing, but model does not expose checkpointing controls.")
        

    def _dynamic_preprocess_tensor(self, image_t, min_num=1, max_num=1, use_thumbnail=False):
        C, orig_height, orig_width = image_t.shape
        
        target_aspect_ratio = get_target_aspect_ratio(
            orig_width, orig_height, self.image_size, min_num, max_num)
            
        ratio_w, ratio_h = target_aspect_ratio[0], target_aspect_ratio[1]
        target_width = self.image_size * ratio_w
        target_height = self.image_size * ratio_h
        blocks = ratio_w * ratio_h
        
        resized_img = TF.resize(
            image_t, 
            size=[target_height, target_width], 
            interpolation=InterpolationMode.BICUBIC, 
            antialias=True
        )
        
        
        # 1. view to -> [C, ratio_h, image_size, ratio_w, image_size]
        reshaped = resized_img.view(C, ratio_h, self.image_size, ratio_w, self.image_size)
        # 2. permute to -> [ratio_h, ratio_w, C, image_size, image_size]
        permuted = reshaped.permute(1, 3, 0, 2, 4)
        # 3. reshape to -> [blocks, C, image_size, image_size]
        stacked_tiles = permuted.reshape(blocks, C, self.image_size, self.image_size)
            
        if use_thumbnail and blocks != 1:
            thumbnail_img = TF.resize(
                image_t, 
                size=[self.image_size, self.image_size], 
                interpolation=InterpolationMode.BICUBIC, 
                antialias=True
            )
            stacked_tiles = torch.cat([stacked_tiles, thumbnail_img.unsqueeze(0)], dim=0)
            
        return stacked_tiles

    def _preprocess_images(
        self,
        image_tensors_batch: List[List[Union[Image.Image, torch.Tensor]]]
    ) -> tuple[torch.Tensor, List[List[int]]]:

        pixel_values_list = []
        batch_num_tiles_list = []
        
        dtype = torch.bfloat16
        mean = torch.tensor(IMAGENET_MEAN, device=self.device, dtype=dtype).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD, device=self.device, dtype=dtype).view(1, 3, 1, 1)

        for image_tensors in image_tensors_batch:
            num_tiles_list = []
            for image in image_tensors:
                if isinstance(image, torch.Tensor):
                    image_t = image.to(self.device, dtype=torch.float32)
                    if image_t.ndim == 3 and image_t.shape[1] == self.image_size and image_t.shape[2] == self.image_size:
                        tile_tensors = (image_t.unsqueeze(0).to(dtype=dtype) - mean) / std
                    else:
                        tiles = self._dynamic_preprocess_tensor(image_t).to(dtype=dtype)
                        tile_tensors = (tiles - mean) / std
                else:
                    image_t = TF.to_tensor(image).to(self.device, dtype=torch.float32)
                    tiles = self._dynamic_preprocess_tensor(image_t).to(dtype=dtype)
                    tile_tensors = (tiles - mean) / std
                
                pixel_values_list.append(tile_tensors)
                num_tiles_list.append(tile_tensors.shape[0])
            batch_num_tiles_list.append(num_tiles_list)

        if len(pixel_values_list) > 0:
            pixel_values = torch.cat(pixel_values_list, dim=0)
        else:
            pixel_values = torch.empty(0, 3, self.image_size, self.image_size, dtype=torch.bfloat16, device=self.device)

        return pixel_values, batch_num_tiles_list

    def _build_multimodal_prompt(
        self,
        batch_num_tiles_list: List[List[int]],
        text_prompts: List[str]
    ) -> List[str]:

        IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"
        IMG_START_TOKEN = "<img>"
        IMG_END_TOKEN = "</img>"
        self.img_context_token_id = self.tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)

        prompts = []
        for num_tiles_list, text_prompt in zip(batch_num_tiles_list, text_prompts):
            prompt_segments = []
            for i, tile_count in enumerate(num_tiles_list):
                token_count = self.model.num_image_token * tile_count
                image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * token_count + IMG_END_TOKEN
                prompt_segments.append(f"Image-{i+1}: {image_tokens}\n")
            
            final_prompt = "".join(prompt_segments) + text_prompt.strip()
            prompts.append(final_prompt)

        return prompts
    
    def _prepare_and_fuse_embeddings(
        self,
        prompts: List[str],
        vit_embeds: torch.Tensor,
        image_masks: torch.Tensor,
        batch_num_tiles_list: List[List[int]]
    ) -> tuple[torch.Tensor, torch.Tensor]:
   
        untruncated_ids = self.tokenizer(prompts, padding=False, truncation=False)["input_ids"]
        true_sequence_length = max(len(ids) for ids in untruncated_ids) if len(untruncated_ids) > 0 else 0

        if true_sequence_length > self.max_text_length:
            print("\n" + "="*80)
            print(f" WARNING: Input prompt was TRUNCATED in batch!")
            print(f"   - Max Length Allowed    : {self.max_text_length}")
            print(f"   - Actual Max Length     : {true_sequence_length}")
            print("="*80 + "\n")

        model_inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding='max_length',
            truncation=True,
            max_length=self.max_text_length
        ).to(self.device)
        input_ids = model_inputs["input_ids"]
        attention_mask = model_inputs["attention_mask"]

        img_token_mask = (input_ids == self.img_context_token_id)
        input_embeds = self.model.language_model.get_input_embeddings()(input_ids).clone()

        B, N, C = input_embeds.shape
        vit_embeds = vit_embeds.reshape(-1, C)
        
        tokens_per_tile = self.model.num_image_token 
        vit_idx = 0
 
        actual_vis_tokens_list = img_token_mask.sum(dim=1).tolist()
            
        for b in range(B):
            expected_vis_tokens = sum(batch_num_tiles_list[b]) * tokens_per_tile
            mask_b = img_token_mask[b]
            actual_vis_tokens = actual_vis_tokens_list[b]
            
            item_vit_embeds = vit_embeds[vit_idx : vit_idx + expected_vis_tokens]
            vit_idx += expected_vis_tokens
            
            if actual_vis_tokens > 0:
                input_embeds[b, mask_b] = item_vit_embeds[:actual_vis_tokens]
            
            current_token_idx = 0
            img_token_locations = torch.where(mask_b)[0]
            
            for i in range(len(batch_num_tiles_list[b])):
                num_tiles_for_this_image = batch_num_tiles_list[b][i]
                num_tokens_for_this_image = num_tiles_for_this_image * tokens_per_tile
           
                if not image_masks[b][i]:
                    start_offset = current_token_idx
                    end_offset = min(current_token_idx + num_tokens_for_this_image, len(img_token_locations))
                    if start_offset < end_offset:
                        idxs = img_token_locations[start_offset:end_offset]
                        attention_mask[b, idxs] = 0
        
                current_token_idx += num_tokens_for_this_image
     
        return input_embeds, attention_mask

    def get_fused_image_text_embedding_from_tensor_images(
        self,
        image_tensors_batch: List[List[Union[Image.Image, torch.Tensor]]],
        image_masks: torch.Tensor,
        text_prompts: List[str],
        return_cls_only: bool = True,
    ):

        pixel_values, batch_num_tiles_list = self._preprocess_images(image_tensors_batch)

        if pixel_values.shape[0] == 0:
            print("Warning: No valid images to process after masking.")
            vit_embeds = torch.empty(0, self.model.config.hidden_size).to(self.device, dtype=torch.bfloat16)
        else:
            vit_embeds = self.model.extract_feature(pixel_values)
        
        prompts = self._build_multimodal_prompt(batch_num_tiles_list, text_prompts)
        inputs_embeds, attention_mask = self._prepare_and_fuse_embeddings(prompts, vit_embeds, image_masks, batch_num_tiles_list)

        outputs = self.model.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
        fused_hidden = outputs.logits if hasattr(outputs, "logits") else outputs[0]

        return fused_hidden[:, 0, :] if return_cls_only else fused_hidden
