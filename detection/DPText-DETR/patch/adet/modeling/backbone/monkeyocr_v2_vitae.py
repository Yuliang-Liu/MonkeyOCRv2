import json
import logging
import os
from collections import OrderedDict
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from detectron2.layers import Conv2d, ShapeSpec, get_norm
from detectron2.modeling import BACKBONE_REGISTRY, Backbone


logger = logging.getLogger(__name__)


class MonkeyOCRV2ViTAEBackbone(Backbone):
    """Expose MonkeyOCRv2-ViTAE stage2/3/4 maps as DPText-DETR res3/4/5."""

    def __init__(self, cfg, input_shape: ShapeSpec):
        super().__init__()
        self.device = torch.device(cfg.MODEL.DEVICE)
        vitae_cfg = cfg.MODEL.MONKEY_V2_VITAE
        self.model_dir = vitae_cfg.MODEL_DIR
        self.weights_path = vitae_cfg.WEIGHTS
        if not self.model_dir:
            raise ValueError("MODEL.MONKEY_V2_VITAE.MODEL_DIR must point to the MonkeyOCRv2-ViTAE model directory.")

        try:
            from safetensors.torch import load_file as load_safetensors_file
            from transformers import AutoConfig, AutoModel
        except ImportError as exc:
            raise ImportError("MonkeyOCRv2-ViTAE backbone requires `transformers` and `safetensors`.") from exc

        trust_remote_code = bool(vitae_cfg.TRUST_REMOTE_CODE)
        if self.weights_path and self.weights_path.endswith(".safetensors"):
            hf_cfg = AutoConfig.from_pretrained(self.model_dir, trust_remote_code=trust_remote_code)
            self.vitae = AutoModel.from_config(hf_cfg, trust_remote_code=trust_remote_code)
            state_dict = load_safetensors_file(self.weights_path)
            missing, unexpected = self.vitae.load_state_dict(state_dict, strict=False)
            if missing or unexpected:
                raise RuntimeError(
                    f"Failed to load MonkeyOCRv2-ViTAE weights from {self.weights_path}; "
                    f"missing={missing}, unexpected={unexpected}"
                )
            logger.info("Loaded MonkeyOCRv2-ViTAE weights from safetensors: %s", self.weights_path)
        else:
            self.vitae = AutoModel.from_pretrained(
                self.weights_path or self.model_dir,
                trust_remote_code=trust_remote_code,
            )
            logger.info("Loaded MonkeyOCRv2-ViTAE weights from: %s", self.weights_path or self.model_dir)

        self.vitae.gradient_checkpointing = bool(vitae_cfg.GRADIENT_CHECKPOINTING)
        if hasattr(self.vitae.config, "gradient_checkpointing"):
            self.vitae.config.gradient_checkpointing = bool(vitae_cfg.GRADIENT_CHECKPOINTING)

        model_dtype = torch.bfloat16 if bool(vitae_cfg.USE_BF16) else torch.float32
        self.vitae.to(device=self.device, dtype=model_dtype)

        processor_cfg_path = os.path.join(self.model_dir, "preprocessor_config.json")
        with open(processor_cfg_path, "r", encoding="utf-8") as f:
            processor_cfg = json.load(f)

        self.image_mean = torch.tensor(
            processor_cfg.get("image_mean", [0.48145466, 0.4578275, 0.40821073]),
            dtype=torch.float32,
        ).view(1, -1, 1, 1)
        self.image_std = torch.tensor(
            processor_cfg.get("image_std", [0.26862954, 0.26130258, 0.27577711]),
            dtype=torch.float32,
        ).view(1, -1, 1, 1)
        self.patch_size = int(processor_cfg.get("patch_size", getattr(self.vitae.config, "patch_size", 32)))
        self.temporal_patch_size = int(
            processor_cfg.get("temporal_patch_size", getattr(self.vitae.config, "temporal_patch_size", 1))
        )
        self.merge_size = int(processor_cfg.get("merge_size", 1))
        if self.temporal_patch_size != 1:
            raise ValueError(f"Only temporal_patch_size=1 is supported, got {self.temporal_patch_size}.")
        if self.merge_size != 1:
            raise ValueError(f"MonkeyOCRv2-ViTAE detection path expects merge_size=1, got {self.merge_size}.")

        stage_dims = list(getattr(self.vitae.config, "stage_dims", [64, 128, 256, 512]))
        out_features = list(vitae_cfg.OUT_FEATURES)
        if out_features != ["res3", "res4", "res5"]:
            raise ValueError(f"Only OUT_FEATURES ['res3', 'res4', 'res5'] are supported, got {out_features}.")

        native_channels = {"res3": int(stage_dims[1]), "res4": int(stage_dims[2]), "res5": int(stage_dims[3])}
        requested_out_channels = int(vitae_cfg.OUT_CHANNELS)
        self.proj = None
        if requested_out_channels > 0:
            self.proj = nn.ModuleDict(
                {
                    name: Conv2d(
                        native_channels[name],
                        requested_out_channels,
                        kernel_size=1,
                        bias=False,
                        norm=get_norm("LN", requested_out_channels),
                    )
                    for name in out_features
                }
            )
            self._out_feature_channels = {name: requested_out_channels for name in out_features}
        else:
            self._out_feature_channels = native_channels

        self._out_features = out_features
        self._out_feature_strides = {"res3": 8, "res4": 16, "res5": 32}
        self._size_divisibility = self.patch_size
        self.uses_external_processor = True
        self.expects_image_list = True

        if bool(vitae_cfg.FREEZE):
            for p in self.vitae.parameters():
                p.requires_grad = False
            self.vitae.eval()

    def _to_flatten_patches(self, image: torch.Tensor, valid_size: Tuple[int, int]) -> Tuple[torch.Tensor, List[int]]:
        if image.dim() != 3:
            raise ValueError(f"Expected CHW image tensor, got shape={tuple(image.shape)}")
        image = image.to(self.device)
        if image.dtype != torch.uint8:
            image = image.clamp(0, 255).to(torch.uint8)

        c, h, w = image.shape
        if c != 3:
            raise ValueError(f"Expected 3-channel RGB tensor, got channels={c}.")
        if h % self.patch_size != 0 or w % self.patch_size != 0:
            raise ValueError(
                f"Input image size {(h, w)} must be divisible by patch_size={self.patch_size}. "
                "Check backbone size_divisibility and SMART_RESIZE.FACTOR."
            )

        image_f = image.to(torch.float32).unsqueeze(0) / 255.0
        mean = self.image_mean.to(device=self.device)
        std = self.image_std.to(device=self.device)
        image_f = (image_f - mean) / std

        # pad 0 to the invalid regions after normalization
        h_valid, w_valid = valid_size
        if h_valid < h:
            image_f[:, :, h_valid:, :] = 0.0
        if w_valid < w:
            image_f[:, :, :, w_valid:] = 0.0

        grid_t = 1
        grid_h = h // self.patch_size
        grid_w = w // self.patch_size
        patches = image_f.reshape(
            grid_t,
            self.temporal_patch_size,
            c,
            grid_h,
            self.patch_size,
            grid_w,
            self.patch_size,
        )
        patches = patches.permute(0, 3, 5, 2, 1, 4, 6)
        flatten = patches.reshape(
            grid_t * grid_h * grid_w,
            c * self.temporal_patch_size * self.patch_size * self.patch_size,
        )
        return flatten, [grid_t, grid_h, grid_w]

    @staticmethod
    def _split_stage_tokens(tokens: torch.Tensor, grid_hw: torch.Tensor) -> List[torch.Tensor]:
        maps = []
        offset = 0
        for i in range(grid_hw.shape[0]):
            h = int(grid_hw[i, 0].item())
            w = int(grid_hw[i, 1].item())
            n = h * w
            cur = tokens[offset : offset + n]
            maps.append(cur.view(h, w, -1).permute(2, 0, 1).contiguous())
            offset += n
        return maps

    @staticmethod
    def _pad_feature_maps(maps: List[torch.Tensor]) -> torch.Tensor:
        if not maps:
            raise ValueError("Expected non-empty feature map list.")
        first_shape = maps[0].shape
        if all(feat.shape == first_shape for feat in maps):
            return torch.stack(maps, dim=0)

        max_h = max(x.shape[-2] for x in maps)
        max_w = max(x.shape[-1] for x in maps)
        c = maps[0].shape[0]
        batch = maps[0].new_zeros((len(maps), c, max_h, max_w))
        for i, feat in enumerate(maps):
            h, w = feat.shape[-2:]
            batch[i, :, :h, :w] = feat
        return batch

    def forward(self, images) -> Dict[str, torch.Tensor]:
        pixel_values_list = []
        image_grid_thw_list = []
        valid_sizes = getattr(images, "valid_sizes", images.image_sizes)
        for idx in range(len(images.image_sizes)):
            flatten_patches, image_grid_thw = self._to_flatten_patches(images.tensor[idx], valid_sizes[idx])
            pixel_values_list.append(flatten_patches)
            image_grid_thw_list.append(image_grid_thw)

        pixel_values = torch.cat(pixel_values_list, dim=0).to(self.device)
        image_grid_thw = torch.tensor(image_grid_thw_list, dtype=torch.long, device=self.device)
        pixel_values = pixel_values.to(dtype=next(self.vitae.parameters()).dtype)

        vision_embeddings_all, grid_hw_all = self.vitae(pixel_values, image_grid_thw)
        stage_to_name = [(1, "res3"), (2, "res4"), (3, "res5")]
        outputs = OrderedDict()
        for stage_idx, name in stage_to_name:
            maps = self._split_stage_tokens(vision_embeddings_all[stage_idx], grid_hw_all[stage_idx])
            feat = self._pad_feature_maps(maps)
            if self.proj is not None:
                feat = self.proj[name](feat.to(dtype=self.proj[name].weight.dtype))
            else:
                feat = feat.float()
            outputs[name] = feat
        return outputs

    def output_shape(self):
        return {
            name: ShapeSpec(channels=self._out_feature_channels[name], stride=self._out_feature_strides[name])
            for name in self._out_features
        }

    @property
    def size_divisibility(self):
        return self._size_divisibility


@BACKBONE_REGISTRY.register()
def build_monkeyocr_v2_vitae_backbone(cfg, input_shape: ShapeSpec):
    return MonkeyOCRV2ViTAEBackbone(cfg, input_shape)
