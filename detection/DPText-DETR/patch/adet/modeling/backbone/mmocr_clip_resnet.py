"""Detectron2 wrapper around mmocr.models.common.backbones.clip_resnet.CLIPResNet.

This exposes the oCLIP ResNet-50 (deep stem + CLIPBottleneck with avgpool
plugins) as a standard :class:`detectron2.modeling.Backbone` so that it can
be plugged into DPText-DETR's ``TransformerPureDetector`` meta arch with no
other changes.

The wrapper is *self-contained*: it sets ``uses_external_processor = True``
and ``expects_image_list = True`` so the meta arch forwards the raw
``ImageList`` (RGB uint8) to the backbone, leaving normalisation to the
forward method.  We use the **oCLIP/CLIP mean/std in the [0, 255]
domain** (taken verbatim from
``bytedance/oclip`` ``src/clip/clip.py:68``) so that the input
distribution exactly matches the one the official
``resnet50-oclip-7ba0c533.pth`` was pretrained on.  This is critical
because ``Base.yaml`` sets ``MMOCR_CLIP_RESNET.FROZEN_STAGES=1`` which
freezes the stem and layer1 BN statistics; with the original ImageNet
mean/std those frozen BN statistics (recorded for a std≈0.85 input
distribution) see a 1.17× larger distribution and never get the chance
to adapt, slowing down convergence.  Matching oCLIP's mean/std lets the
frozen BN statistics stay valid and removes the bias.  The first
convolution ``stem.0`` is kept as the original ``Conv2d(3, 32, 3)`` so
the official ``resnet50-oclip-7ba0c533.pth`` weights load cleanly with
no conversion (``state_dict`` keys match
``mmocr.models.common.backbones.clip_resnet`` exactly).
"""
from __future__ import annotations

import logging
import os
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from detectron2.layers import ShapeSpec
from detectron2.modeling import BACKBONE_REGISTRY, Backbone

logger = logging.getLogger(__name__)


# oCLIP/CLIP normalisation in the [0, 255] domain (RGB).  These are the
# mean/std used by ``bytedance/oclip`` at training time
# (``src/clip/clip.py:68``, where they are written in [0, 1] domain and
# then applied after ``ToTensor()``):
#   mean([0,1]) = (0.48145466, 0.4578275,  0.40821073)
#   std ([0,1]) = (0.26862954, 0.26130258, 0.27577711)
# Using these values (instead of the standard ImageNet ones) makes the
# wrapper's forward pass match the distribution the oCLIP backbone was
# pretrained on, so the frozen stem/layer1 BN running statistics
# (recorded for this exact distribution) stay valid.  The values below
# are the originals multiplied by 255 (not rounded further) so the
# resulting 1/0 differences do not bias the BN normalisation.
_OCLIP_MEAN_RGB = (0.48145466 * 255, 0.4578275 * 255, 0.40821073 * 255)
_OCLIP_STD_RGB = (0.26862954 * 255, 0.26130258 * 255, 0.27577711 * 255)


class MMOCRCLIPResNetBackbone(Backbone):
    """Wrap mmocr's CLIPResNet as a detectron2 Backbone.

    Forwards a tuple of (res2, res3, res4, res5) features (after a small 1x1
    stride-1 channel adapter on res3/res4/res5 if ``OUT_CHANNELS`` is set).
    The output feature names match what DPText-DETR expects from
    ``build_resnet_backbone``: ``res3``, ``res4``, ``res5`` with strides
    8/16/32.  The optional res2 (stride 4) feature is exposed as
    ``OUT_FEATURES2`` so it can be used by FPN variants later.
    """

    def __init__(self, cfg, input_shape: ShapeSpec):
        super().__init__()
        self.device = torch.device(cfg.MODEL.DEVICE)

        # Pull CLIPResNet from mmocr.  Importing the module registers the
        # backbone with the mmocr ``MODELS`` registry; we instantiate the
        # class directly to avoid going through the mmengine config system.
        from mmocr.models.common.backbones.clip_resnet import CLIPResNet
        from mmocr.registry import MODELS  # noqa: F401  (ensures registry init)

        backbone_cfg = cfg.MODEL.MMOCR_CLIP_RESNET
        # mmocr's CLIPResNet accepts the same kwargs as mmdet's ResNet (see
        # mmdet/models/backbones/resnet.py).  We disable the mmengine
        # ``init_cfg`` path (AdetCheckpointer loads the oCLIP weights from
        # ``MODEL.WEIGHTS`` afterwards).
        self.net = CLIPResNet(
            depth=int(backbone_cfg.DEPTH),
            strides=tuple(backbone_cfg.STRIDES),
            deep_stem=bool(backbone_cfg.DEEP_STEM),
            avg_down=bool(backbone_cfg.AVG_DOWN),
            frozen_stages=int(backbone_cfg.FROZEN_STAGES),
            conv_cfg=None,
            norm_cfg=dict(type=backbone_cfg.NORM_TYPE, requires_grad=True),
            norm_eval=bool(backbone_cfg.NORM_EVAL),
            with_cp=bool(backbone_cfg.WITH_CP),
            zero_init_residual=bool(backbone_cfg.ZERO_INIT_LAST_BN),
            init_cfg=None,
        )

        # Pre-compute (1, 3, 1, 1) tensors for the oCLIP normalisation.
        # DPText-DETR loads images in RGB (Base.yaml sets FORMAT: "RGB"), so
        # we use the mean/std directly without channel swapping.
        mean_rgb = torch.tensor(
            _OCLIP_MEAN_RGB, dtype=torch.float32
        ).view(1, 3, 1, 1)
        std_rgb = torch.tensor(
            _OCLIP_STD_RGB, dtype=torch.float32
        ).view(1, 3, 1, 1)
        self.register_buffer("_mean_rgb", mean_rgb, persistent=False)
        self.register_buffer("_std_rgb", std_rgb, persistent=False)

        # Output feature configuration.  mmocr's CLIPResNet always returns
        # (res2, res3, res4, res5) at strides 4, 8, 16, 32; we keep the
        # three that DPText-DETR's transformer consumes.
        out_features = list(backbone_cfg.OUT_FEATURES)
        if out_features != ["res3", "res4", "res5"]:
            raise ValueError(
                f"Only OUT_FEATURES ['res3', 'res4', 'res5'] are supported, got {out_features}."
            )
        self._out_features = out_features
        self._out_feature_strides = {"res3": 8, "res4": 16, "res5": 32}
        self._out_feature_channels = {"res3": 512, "res4": 1024, "res5": 2048}
        self._size_divisibility = 32

        # Signal to the meta arch that the backbone wants the raw ImageList
        # and will handle its own normalisation.
        self.uses_external_processor = True
        self.expects_image_list = True

        # Move to target device/dtype.
        dtype = torch.bfloat16 if bool(backbone_cfg.USE_BF16) else torch.float32
        self.net.to(device=self.device, dtype=dtype)
        # The buffers must stay in fp32 to keep the normalisation accurate.
        self._mean_rgb = self._mean_rgb.to(self.device)
        self._std_rgb = self._std_rgb.to(self.device)

    def _normalise(self, image_rgb: torch.Tensor) -> torch.Tensor:
        """RGB uint8 tensor (N, 3, H, W) -> oCLIP-normalised RGB fp tensor.

        Mirrors the standard mmdet/d2 convention: keep the image in the
        0-255 range and subtract the per-channel mean/std in that domain
        (no /255 step).  The mean/std used here are the oCLIP/CLIP values
        (see ``_OCLIP_MEAN_RGB`` / ``_OCLIP_STD_RGB`` and
        ``bytedance/oclip`` ``src/clip/clip.py:68``) so the input
        distribution exactly matches the one the backbone was pretrained
        on; this is required because ``FROZEN_STAGES=1`` freezes the
        stem/layer1 BN statistics.
        """
        if image_rgb.dim() != 4 or image_rgb.shape[1] != 3:
            raise ValueError(
                f"Expected (N, 3, H, W) RGB image, got shape={tuple(image_rgb.shape)}"
            )
        if image_rgb.dtype != torch.uint8:
            image_rgb = image_rgb.clamp(0, 255).to(torch.uint8)
        image = image_rgb.to(torch.float32)  # RGB in [0, 255]
        image = (image - self._mean_rgb) / self._std_rgb
        return image

    def forward(self, images) -> Dict[str, torch.Tensor]:
        if not hasattr(images, "tensor") or not hasattr(images, "image_sizes"):
            raise TypeError(
                "MMOCRCLIPResNetBackbone expects an ImageList-like object; "
                "set `expects_image_list=True` in the meta arch."
            )
        x = self._normalise(images.tensor)
        # Cast to the backbone's parameter dtype (fp32 or bf16).
        net_dtype = next(self.net.parameters()).dtype
        x = x.to(dtype=net_dtype)
        feats: Tuple[torch.Tensor, ...] = self.net(x)
        # mmocr returns (res2, res3, res4, res5).
        out = {
            "res2": feats[0],
            "res3": feats[1],
            "res4": feats[2],
            "res5": feats[3],
        }
        return {k: out[k] for k in self._out_features}

    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name],
                stride=self._out_feature_strides[name],
            )
            for name in self._out_features
        }

    @property
    def size_divisibility(self) -> int:
        return self._size_divisibility

    @property
    def padding_constraints(self) -> dict:
        return {"size_divisibility": self._size_divisibility, "square": False}


@BACKBONE_REGISTRY.register()
def build_mmocr_clip_resnet_backbone(cfg, input_shape: ShapeSpec):
    return MMOCRCLIPResNetBackbone(cfg, input_shape)
