# Copyright (c) OpenMMLab. All rights reserved.
"""mmdet-compatible wrapper for the MonkeyOCRv2-ViTAE vision encoder.

The original backbone lives in
``DPText-DETR/adet/modeling/backbone/monkeyocr_v2_vitae.py`` and is wired to
detectron2 (uses ``detectron2.layers`` and ``BACKBONE_REGISTRY``). This file
re-implements the same logic against the mmengine BaseModule interface so that
it can be plugged into mmocr's detection models.

The ViTAE model is a 4-stage ViTAEv2 vision encoder. We expose its four stages
as ``res2``/``res3``/``res4``/``res5`` feature maps (strides 4/8/16/32) so they
match the layout FPNC expects from a ResNet-50.
"""

import json
import logging
import os
from collections import OrderedDict
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from mmcv.cnn import build_norm_layer
from mmengine.model import BaseModule

from mmocr.registry import MODELS

logger = logging.getLogger(__name__)


@MODELS.register_module()
class MonkeyOCRv2ViTAEBackbone(BaseModule):
    """Expose the 4 stages of MonkeyOCRv2-ViTAE as
    ``res2``/``res3``/``res4``/``res5`` (strides 4/8/16/32).

    The backbone owns a HuggingFace ``AutoModel`` loaded from a local
    checkpoint directory. Image normalization (mean/std) and patch splitting
    are performed inside the backbone itself, so the data preprocessor must
    be configured with ``mean=None``, ``std=None`` and a
    ``pad_size_divisor`` equal to (or a multiple of) the ViTAE
    ``patch_size`` (32 by default).

    Args:
        model_dir (str): Local path to the ViTAE model directory
            (containing ``config.json`` and ``model.safetensors``).
        weights (str, optional): Path to the safetensors weights. If it ends
            with ``.safetensors`` the model is loaded with config from
            ``model_dir`` and the weights are loaded from this path. Defaults
            to ``None``, in which case ``model_dir`` is passed to
            ``AutoModel.from_pretrained``.
        out_features (Sequence[str]): Which stages to expose. Must be a subset
            of ``['res2', 'res3', 'res4', 'res5']``. Defaults to all four.
        out_channels (int): If > 0, project each selected stage to this many
            channels via a 1x1 conv + LN. If 0, the native stage channels are
            returned. Defaults to 0.
        use_bf16 (bool): Cast the ViTAE to bfloat16. The projection layers
            stay in float32. Defaults to ``False``.
        trust_remote_code (bool): Forwarded to ``transformers.AutoModel``.
            Required for MonkeyOCRv2-ViTAE. Defaults to ``True``.
        freeze (bool): If ``True``, freeze all ViTAE parameters. Defaults to
            ``False``.
        gradient_checkpointing (bool): Enable gradient checkpointing in the
            ViTAE. Defaults to ``False``.
        norm_cfg (dict): Norm layer config for the projection 1x1 convs.
            Defaults to ``dict(type='LN')``.
        init_cfg (dict or list[dict], optional): mmengine initialization
            config for the projection layers.
    """

    # stage index (0..3) → res-level name
    _STAGE_TO_NAME = ((0, 'res2'), (1, 'res3'), (2, 'res4'), (3, 'res5'))
    _VALID_NAMES = ('res2', 'res3', 'res4', 'res5')

    def __init__(self,
                 model_dir: str,
                 weights: Optional[str] = None,
                 out_features: Sequence[str] = ('res2', 'res3', 'res4',
                                                'res5'),
                 out_channels: int = 0,
                 use_bf16: bool = False,
                 trust_remote_code: bool = True,
                 freeze: bool = False,
                 gradient_checkpointing: bool = False,
                 norm_cfg: dict = dict(type='LN'),
                 init_cfg: Optional[dict] = None) -> None:
        super().__init__(init_cfg=init_cfg)

        if not model_dir:
            raise ValueError(
                'MonkeyOCRv2ViTAEBackbone requires `model_dir` pointing to '
                'the local ViTAE model directory.')

        out_features = tuple(out_features)
        for name in out_features:
            if name not in self._VALID_NAMES:
                raise ValueError(
                    f'Unknown out_features={name!r}; valid options: '
                    f'{self._VALID_NAMES}.')

        try:
            from safetensors.torch import load_file as load_safetensors_file
            from transformers import AutoConfig, AutoModel
        except ImportError as exc:
            raise ImportError(
                'MonkeyOCRv2ViTAEBackbone requires `transformers` and '
                '`safetensors`. Please install them.') from exc

        self.model_dir = model_dir
        self.weights_path = weights
        self.out_features = out_features
        self.use_bf16 = bool(use_bf16)
        self.patch_size = 32
        self.temporal_patch_size = 1
        self.merge_size = 1

        # --- Load ViTAE weights -------------------------------------------------
        if self.weights_path and self.weights_path.endswith('.safetensors'):
            hf_cfg = AutoConfig.from_pretrained(
                self.model_dir, trust_remote_code=trust_remote_code)
            self.vitae = AutoModel.from_config(
                hf_cfg, trust_remote_code=trust_remote_code)
            state_dict = load_safetensors_file(self.weights_path)
            missing, unexpected = self.vitae.load_state_dict(
                state_dict, strict=False)
            if missing or unexpected:
                raise RuntimeError(
                    f'Failed to load MonkeyOCRv2-ViTAE weights from '
                    f'{self.weights_path}; missing={missing}, '
                    f'unexpected={unexpected}')
            logger.info('Loaded MonkeyOCRv2-ViTAE weights from %s',
                        self.weights_path)
        else:
            self.vitae = AutoModel.from_pretrained(
                self.weights_path or self.model_dir,
                trust_remote_code=trust_remote_code)
            logger.info('Loaded MonkeyOCRv2-ViTAE from %s',
                        self.weights_path or self.model_dir)

        # --- Read processor config (mean/std/patch_size) ------------------------
        processor_cfg_path = os.path.join(self.model_dir,
                                          'preprocessor_config.json')
        with open(processor_cfg_path, 'r', encoding='utf-8') as f:
            processor_cfg = json.load(f)

        self.image_mean = torch.tensor(
            processor_cfg.get('image_mean',
                              [0.48145466, 0.4578275, 0.40821073]),
            dtype=torch.float32).view(1, -1, 1, 1)
        self.image_std = torch.tensor(
            processor_cfg.get('image_std',
                              [0.26862954, 0.26130258, 0.27577711]),
            dtype=torch.float32).view(1, -1, 1, 1)
        self.patch_size = int(
            processor_cfg.get('patch_size',
                              getattr(self.vitae.config, 'patch_size', 32)))
        self.temporal_patch_size = int(
            processor_cfg.get(
                'temporal_patch_size',
                getattr(self.vitae.config, 'temporal_patch_size', 1)))
        self.merge_size = int(processor_cfg.get('merge_size', 1))
        if self.temporal_patch_size != 1:
            raise ValueError(
                f'Only temporal_patch_size=1 is supported, got '
                f'{self.temporal_patch_size}.')
        if self.merge_size != 1:
            raise ValueError(
                f'MonkeyOCRv2-ViTAE detection path expects merge_size=1, got '
                f'{self.merge_size}.')

        # --- Enable gradient checkpointing -------------------------------------
        self.vitae.gradient_checkpointing = bool(gradient_checkpointing)
        if hasattr(self.vitae.config, 'gradient_checkpointing'):
            self.vitae.config.gradient_checkpointing = bool(
                gradient_checkpointing)

        # --- Native stage dimensions -------------------------------------------
        stage_dims = list(
            getattr(self.vitae.config, 'stage_dims', [64, 128, 256, 512]))
        if len(stage_dims) != 4:
            raise ValueError(
                f'Expected stage_dims of length 4, got {stage_dims}.')
        self._native_channels = {
            'res2': int(stage_dims[0]),
            'res3': int(stage_dims[1]),
            'res4': int(stage_dims[2]),
            'res5': int(stage_dims[3]),
        }
        self._out_feature_strides = {
            'res2': 4,
            'res3': 8,
            'res4': 16,
            'res5': 32,
        }

        # --- Optional 1x1 + LN projection --------------------------------------
        self.proj = None
        if int(out_channels) > 0:
            proj_modules = OrderedDict()
            for name in self.out_features:
                proj_modules[name] = nn.Sequential(
                    nn.Conv2d(self._native_channels[name],
                              int(out_channels),
                              kernel_size=1,
                              bias=False),
                    build_norm_layer(norm_cfg, int(out_channels))[1],
                )
            self.proj = nn.ModuleDict(proj_modules)
            self._out_feature_channels = {
                name: int(out_channels) for name in self.out_features
            }
        else:
            self._out_feature_channels = {
                name: self._native_channels[name] for name in self.out_features
            }

        # Cast before DistributedDataParallel is constructed. Changing
        # parameter dtype/storage lazily in ``forward`` violates DDP's
        # expectation that parameters remain unchanged after construction and
        # can make replicas drift apart. Projection layers, when enabled,
        # intentionally stay in float32.
        if self.use_bf16:
            self.vitae.to(dtype=torch.bfloat16)

        # --- Freeze ViTAE if requested -----------------------------------------
        if freeze:
            for p in self.vitae.parameters():
                p.requires_grad = False
            self.vitae.eval()

    # ------------------------------------------------------------------ utils
    def init_weights(self) -> None:
        # Override `BaseModule.init_weights` to skip the ViTAE child. By default
        # mmengine's recursive init flow would call
        # `self.vitae.init_weights()` (the HF `PreTrainedModel` method), which
        # in turn calls `initialize_weights()` and re-initializes every
        # parameter, destroying the safetensors we just loaded. We can't simply
        # set `self.vitae.is_init = True` because `is_init` is a property
        # defined on `BaseModule`, not on `PreTrainedModel`, so
        # `getattr(self.vitae, 'is_init', False)` always returns False and the
        # guard is bypassed. Skipping the recursion here is the only reliable
        # fix. The optional `self.proj` layers are pure `nn.Conv2d` + LN and
        # are already initialized by PyTorch's default in their `__init__`,
        # so they need nothing more.
        for m in self.children():
            if m is self.vitae:
                continue
            if hasattr(m, 'init_weights') and not getattr(
                    m, 'is_init', False):
                m.init_weights()
        self._is_init = True

    def train(self, mode: bool = True):
        super().train(mode)
        # Keep ViTAE in eval mode if it is frozen (e.g. BN running stats).
        if not any(p.requires_grad for p in self.vitae.parameters()):
            self.vitae.eval()
        return self

    def _patchify(self, image: torch.Tensor) -> Tuple[torch.Tensor, List[int]]:
        """Convert one CHW uint8 image into ViTAE's flattened patch tokens.

        Args:
            image (Tensor): ``(3, H, W)`` uint8 image (RGB order).

        Returns:
            flatten (Tensor): ``(T*H*W, C*t*ph*pw)`` flattened patch tokens.
            grid (list[int]): ``[T, H, W]`` patch grid.
        """
        if image.dim() != 3:
            raise ValueError(
                f'Expected CHW image tensor, got shape={tuple(image.shape)}')
        if image.dtype != torch.uint8:
            image = image.clamp(0, 255).to(torch.uint8)
        c, h, w = image.shape
        if c != 3:
            raise ValueError(f'Expected 3-channel RGB tensor, got {c}.')

        if h % self.patch_size != 0 or w % self.patch_size != 0:
            raise ValueError(
                f'Input image size {(h, w)} must be divisible by '
                f'patch_size={self.patch_size}. Ensure the data preprocessor '
                f'pads to a multiple of {self.patch_size}.')

        # Normalize to [0,1] then apply ViTAE mean/std.
        image_f = image.to(torch.float32).unsqueeze(0) / 255.0
        mean = self.image_mean.to(device=image.device)
        std = self.image_std.to(device=image.device)
        image_f = (image_f - mean) / std

        grid_t = 1
        grid_h = h // self.patch_size
        grid_w = w // self.patch_size
        patches = image_f.reshape(grid_t, self.temporal_patch_size, c,
                                  grid_h, self.patch_size, grid_w,
                                  self.patch_size)
        patches = patches.permute(0, 3, 5, 2, 1, 4, 6)
        flatten = patches.reshape(
            grid_t * grid_h * grid_w,
            c * self.temporal_patch_size * self.patch_size * self.patch_size)
        return flatten, [grid_t, grid_h, grid_w]

    @staticmethod
    def _split_stage_tokens(tokens: torch.Tensor,
                            grid_hw: torch.Tensor) -> List[torch.Tensor]:
        maps = []
        offset = 0
        for i in range(grid_hw.shape[0]):
            h = int(grid_hw[i, 0].item())
            w = int(grid_hw[i, 1].item())
            n = h * w
            cur = tokens[offset:offset + n]
            maps.append(cur.view(h, w, -1).permute(2, 0, 1).contiguous())
            offset += n
        return maps

    @staticmethod
    def _pad_feature_maps(maps: List[torch.Tensor]) -> torch.Tensor:
        if not maps:
            raise ValueError('Expected non-empty feature map list.')
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

    # ---------------------------------------------------------------- forward
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Run the ViTAE on a batch of images.

        Args:
            x (Tensor): ``(B, 3, H, W)`` image batch. Channel order must be
                RGB (i.e. configure the data preprocessor with
                ``bgr_to_rgb=True``). The image must be a multiple of
                ``patch_size`` along H and W. Pixel values can be uint8
                (preferred) or float in [0, 255]; the backbone will normalize
                internally using ViTAE's mean/std.

        Returns:
            list[Tensor]: feature maps in the order of ``out_features``
            (default: ``res2``/``res3``/``res4``/``res5``). Each map has
            shape ``(B, C_i, H_i, W_i)``.
        """
        if x.dim() != 4 or x.shape[1] != 3:
            raise ValueError(
                f'Expected (B, 3, H, W) input, got {tuple(x.shape)}.')
        if x.dtype not in (torch.uint8, torch.float32, torch.float16,
                           torch.bfloat16):
            x = x.float()

        device = next(self.vitae.parameters()).device
        vitae_dtype = next(self.vitae.parameters()).dtype

        b, _, h, w = x.shape
        if h % self.patch_size != 0 or w % self.patch_size != 0:
            raise ValueError(
                f'Input spatial size {(h, w)} must be divisible by '
                f'patch_size={self.patch_size}. Configure the data '
                f"preprocessor with pad_size_divisor={self.patch_size}.")

        pixel_values_list = []
        grid_thw_list = []
        for idx in range(b):
            flatten, grid = self._patchify(x[idx])
            pixel_values_list.append(flatten)
            grid_thw_list.append(grid)

        pixel_values = torch.cat(pixel_values_list, dim=0).to(
            device=device, dtype=vitae_dtype)
        image_grid_thw = torch.tensor(grid_thw_list,
                                      dtype=torch.long,
                                      device=device)

        vision_embeddings_all, grid_hw_all = self.vitae(
            pixel_values, image_grid_thw)

        outputs = []
        for stage_idx, name in self._STAGE_TO_NAME:
            if name not in self.out_features:
                continue
            maps = self._split_stage_tokens(vision_embeddings_all[stage_idx],
                                             grid_hw_all[stage_idx])
            feat = self._pad_feature_maps(maps)
            if self.proj is not None:
                proj_module = self.proj[name]
                feat = proj_module(feat.to(dtype=proj_module[0].weight.dtype))
            else:
                feat = feat.float()
            outputs.append(feat)
        return outputs

    # ------------------------------------------------------------- meta info
    def output_shape(self):
        return {
            name: dict(channels=self._out_feature_channels[name],
                       stride=self._out_feature_strides[name])
            for name in self.out_features
        }
