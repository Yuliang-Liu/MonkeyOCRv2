from .mmocr_clip_resnet import build_mmocr_clip_resnet_backbone
from .monkeyocr_v2_vitae import build_monkeyocr_v2_vitae_backbone

__all__ = [
    "build_mmocr_clip_resnet_backbone",
    "build_monkeyocr_v2_vitae_backbone",
]
