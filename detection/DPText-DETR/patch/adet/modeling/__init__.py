# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from .backbone import (
    build_mmocr_clip_resnet_backbone,
    build_monkeyocr_v2_vitae_backbone,
)
from .transformer_detector import TransformerPureDetector

_EXCLUDE = {"torch", "ShapeSpec"}
__all__ = [k for k in globals().keys() if k not in _EXCLUDE and not k.startswith("_")]
