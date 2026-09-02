# Copyright (c) OpenMMLab. All rights reserved.
from .clip_resnet import CLIPResNet
from .monkeyocr_v2_vitae import MonkeyOCRv2ViTAEBackbone
from .unet import UNet

__all__ = ['UNet', 'CLIPResNet', 'MonkeyOCRv2ViTAEBackbone']
