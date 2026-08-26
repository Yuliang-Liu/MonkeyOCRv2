import torch
import torch.nn as nn
from mmengine.model import BaseModule

from mmseg.registry import MODELS
from .timm_dct import FPH, SCSEModule
from .vitaev2 import ViTAEv2


@MODELS.register_module()
class ViTAEv2Dct(BaseModule):
    """ViTAEv2 visual backbone with the original FFDN DCT fusion logic."""

    def __init__(
        self,
        pretrained_checkpoint=None,
        in_channels=3,
        out_indices=(0, 1, 2, 3),
        fusion='ZERO',
        init_cfg=None,
        **kwargs,
    ):
        super().__init__(init_cfg)
        self.out_indices = out_indices
        self.vitae = ViTAEv2(
            in_chans=in_channels,
            out_indices=out_indices,
            pretrained_checkpoint=pretrained_checkpoint,
            init_cfg=None,
            **kwargs,
        )

        self.fph = FPH()
        self.fusion = fusion
        if fusion != 'ZERO':
            raise NotImplementedError('ViTAEv2Dct currently keeps the original ZERO fusion only.')

        stage_dims = self.vitae.tokens_dims
        if len(self.out_indices) < 2:
            raise ValueError('ViTAEv2Dct expects out_indices to include stage 1 for DCT fusion.')
        self.fusion_stage_idx = self.out_indices[1]
        self.fusion_channels = stage_dims[self.fusion_stage_idx]

        self.f_dct_proj = nn.Sequential(
            nn.Conv2d(256, self.fusion_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.fusion_channels),
            nn.ReLU(True),
        )
        self.fusion_pre = nn.Sequential(
            SCSEModule(self.fusion_channels * 2),
            nn.Conv2d(
                self.fusion_channels * 2,
                self.fusion_channels,
                kernel_size=3,
                stride=1,
                padding=1),
            nn.BatchNorm2d(self.fusion_channels),
            nn.ReLU(True),
        )
        self.fusion_post = nn.Conv2d(
            self.fusion_channels,
            self.fusion_channels,
            kernel_size=1,
            stride=1,
            padding=0)
        nn.init.zeros_(self.fusion_post.weight)
        if self.fusion_post.bias is not None:
            nn.init.zeros_(self.fusion_post.bias)

    def forward(self, inputs):
        x, dct, qtb = inputs['x'], inputs['dct'], inputs['qtb']
        f_dct = self.fph(dct, qtb)
        f_dct = self.f_dct_proj(f_dct)

        outs = list(self.vitae(x))
        fusion_out_idx = self.out_indices.index(self.fusion_stage_idx)
        stage_feat = outs[fusion_out_idx]
        ext = self.fusion_pre(torch.cat((stage_feat, f_dct), dim=1))
        outs[fusion_out_idx] = self.fusion_post(ext) + stage_feat
        return tuple(outs)
