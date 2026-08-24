from functools import partial
import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_
import numpy as np
import torch.utils.checkpoint as checkpoint
from .vitaev2_modules.NormalCell import NormalCell
from .vitaev2_modules.ReductionCell import ReductionCell

from mmengine.model import BaseModule
from mmengine.logging import MMLogger
from mmengine.runner.checkpoint import CheckpointLoader
from mmseg.registry import MODELS


def _extract_vitaev2_state_dict(checkpoint):
    if not isinstance(checkpoint, dict):
        return checkpoint

    if 'state_dict' in checkpoint:
        checkpoint = checkpoint['state_dict']
    elif 'model' in checkpoint:
        checkpoint = checkpoint['model']
    elif 'ema' in checkpoint:
        checkpoint = checkpoint['ema']

    if not isinstance(checkpoint, dict):
        return checkpoint

    cleaned = {}
    for key, value in checkpoint.items():
        new_key = key
        for prefix in ('module.backbone.', 'backbone.', 'module.'):
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix):]
                break
        cleaned[new_key] = value
    return cleaned


class PatchEmbedding(nn.Module):
    def __init__(self, inter_channel=32, out_channels=48, img_size=None):
        self.img_size = img_size
        self.inter_channel = inter_channel
        self.out_channel = out_channels
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, inter_channel, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(inter_channel),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(inter_channel, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x, size):
        x = self.conv3(self.conv2(self.conv1(x)))
        b, c, h, w = x.shape
        x = x.permute(0, 2, 3, 1).reshape(b, h*w, c)
        return x, (h, w)

    def flops(self, ):
        flops = 0
        flops += 3 * self.inter_channel * self.img_size[0] * self.img_size[1] // 4 * 9
        flops += self.img_size[0] * self.img_size[1] // 4 * self.inter_channel
        flops += self.inter_channel * self.out_channel * self.img_size[0] * self.img_size[1] // 16 * 9
        flops += self.img_size[0] * self.img_size[1] // 16 * self.out_channel
        flops += self.out_channel * self.out_channel * self.img_size[0] * self.img_size[1] // 16
        return flops

class BasicLayer(nn.Module):
    def __init__(self, img_size=224, in_chans=3, embed_dims=64, token_dims=64, downsample_ratios=4, kernel_size=7, RC_heads=1, NC_heads=6, dilations=[1, 2, 3, 4],
                RC_op='cat', RC_tokens_type='performer', NC_tokens_type='transformer', RC_group=1, NC_group=64, NC_depth=2, dpr=0.1, mlp_ratio=4., qkv_bias=True, 
                qk_scale=None, drop=0, attn_drop=0., norm_layer=nn.LayerNorm, class_token=False, gamma=False, init_values=1e-4, SE=False, window_size=7,
                use_checkpoint=False):
        super().__init__()
        self.img_size = img_size
        self.in_chans = in_chans
        self.embed_dims = embed_dims
        self.token_dims = token_dims
        self.downsample_ratios = downsample_ratios
        self.out_size = self.img_size // self.downsample_ratios
        self.RC_kernel_size = kernel_size
        self.RC_heads = RC_heads
        self.NC_heads = NC_heads
        self.dilations = dilations
        self.RC_op = RC_op
        self.RC_tokens_type = RC_tokens_type
        self.RC_group = RC_group
        self.NC_group = NC_group
        self.NC_depth = NC_depth
        self.use_checkpoint = use_checkpoint
        if RC_tokens_type == 'stem':
            self.RC = PatchEmbedding(inter_channel=token_dims//2, out_channels=token_dims, img_size=img_size)
        elif downsample_ratios > 1:
            self.RC = ReductionCell(img_size, in_chans, embed_dims, token_dims, downsample_ratios, kernel_size,
                            RC_heads, dilations, op=RC_op, tokens_type=RC_tokens_type, group=RC_group, gamma=gamma, init_values=init_values, SE=SE)
        else:
            self.RC = nn.Identity()
        self.NC = nn.ModuleList([
            NormalCell(token_dims, NC_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop, attn_drop=attn_drop,
                       drop_path=dpr[i] if isinstance(dpr, list) else dpr, norm_layer=norm_layer, class_token=class_token, group=NC_group, tokens_type=NC_tokens_type,
                       gamma=gamma, init_values=init_values, SE=SE, img_size=img_size // downsample_ratios, window_size=window_size, shift_size=0)
        for i in range(NC_depth)])

    def forward(self, x, size):
        h, w = size
        x, (h, w) = self.RC(x, (h, w))
        for nc in self.NC:
            nc.H = h
            nc.W = w
            if self.use_checkpoint:
                x = checkpoint.checkpoint(nc, x)
            else:
                x = nc(x)
        return x, (h, w)

@MODELS.register_module()
class ViTAEv2(BaseModule):

    def __init__(self,
                img_size=224,
                in_chans=3,
                embed_dims=64,
                token_dims=64,
                downsample_ratios=[4, 2, 2, 2],
                kernel_size=[7, 3, 3, 3],
                RC_heads=[1, 1, 1, 1],
                NC_heads=4,
                dilations=[[1, 2, 3, 4], [1, 2, 3], [1, 2], [1, 2]],
                RC_op='cat',
                RC_tokens_type='window',
                NC_tokens_type='transformer',
                RC_group=[1, 1, 1, 1],
                NC_group=[1, 32, 64, 64],
                NC_depth=[2, 2, 6, 2], 
                mlp_ratio=4.,
                qkv_bias=True, 
                qk_scale=None, 
                drop_rate=0., 
                attn_drop_rate=0., 
                drop_path_rate=0., 
                norm_layer=partial(nn.LayerNorm, eps=1e-6), 
                stages=4,
                window_size=7,
                out_indices=(0, 1, 2, 3),
                frozen_stages=-1,
                use_checkpoint=False,
                load_ema=True,
                pretrained_checkpoint=None,
                init_cfg=None):
        super().__init__(init_cfg)

        self.stages = stages
        self.load_ema = load_ema
        repeatOrNot = (lambda x, y, z=list: x if isinstance(x, z) else [x for _ in range(y)])
        self.embed_dims = repeatOrNot(embed_dims, stages)
        self.tokens_dims = token_dims if isinstance(token_dims, list) else [token_dims * (2 ** i) for i in range(stages)]
        self.downsample_ratios = repeatOrNot(downsample_ratios, stages)
        self.kernel_size = repeatOrNot(kernel_size, stages)
        self.RC_heads = repeatOrNot(RC_heads, stages)
        self.NC_heads = repeatOrNot(NC_heads, stages)
        self.dilaions = repeatOrNot(dilations, stages)
        self.RC_op = repeatOrNot(RC_op, stages)
        self.RC_tokens_type = repeatOrNot(RC_tokens_type, stages)
        self.NC_tokens_type = repeatOrNot(NC_tokens_type, stages)
        self.RC_group = repeatOrNot(RC_group, stages)
        self.NC_group = repeatOrNot(NC_group, stages)
        self.NC_depth = repeatOrNot(NC_depth, stages)
        self.mlp_ratio = repeatOrNot(mlp_ratio, stages)
        self.qkv_bias = repeatOrNot(qkv_bias, stages)
        self.qk_scale = repeatOrNot(qk_scale, stages)
        self.drop = repeatOrNot(drop_rate, stages)
        self.attn_drop = repeatOrNot(attn_drop_rate, stages)
        self.norm_layer = repeatOrNot(norm_layer, stages)
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.use_checkpoint = use_checkpoint
        self.pretrained_checkpoint = pretrained_checkpoint

        depth = np.sum(self.NC_depth)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        Layers = []
        for i in range(stages):
            startDpr = 0 if i==0 else self.NC_depth[i - 1]
            Layers.append(
                BasicLayer(img_size, in_chans, self.embed_dims[i], self.tokens_dims[i], self.downsample_ratios[i],
                self.kernel_size[i], self.RC_heads[i], self.NC_heads[i], self.dilaions[i], self.RC_op[i],
                self.RC_tokens_type[i], self.NC_tokens_type[i], self.RC_group[i], self.NC_group[i], self.NC_depth[i], dpr[startDpr:self.NC_depth[i]+startDpr],
                mlp_ratio=self.mlp_ratio[i], qkv_bias=self.qkv_bias[i], qk_scale=self.qk_scale[i], drop=self.drop[i], attn_drop=self.attn_drop[i],
                norm_layer=self.norm_layer[i], window_size=window_size, use_checkpoint=use_checkpoint)
            )
            img_size = img_size // self.downsample_ratios[i]
            in_chans = self.tokens_dims[i]

        self.layers = nn.ModuleList(Layers)
        self.num_layers = len(Layers)

        # self._freeze_stages()

    def train(self, mode=True):
        """Convert the model into training mode while keep layers freezed."""
        super(ViTAEv2, self).train(mode)
        self._freeze_stages()

    def _freeze_stages(self):

        if self.frozen_stages > 0:
            for i in range(0, self.frozen_stages):
                m = self.layers[i]
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def init_weights(self):
        """Initialize the weights in backbone.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """

        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm) or isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        logger = MMLogger.get_current_instance()
        checkpoint_path = self.pretrained_checkpoint
        if checkpoint_path is None and self.init_cfg is not None:
            checkpoint_path = self.init_cfg.get('checkpoint', None)

        if checkpoint_path is None:
            logger.warn(f'No pre-trained weights for '
                        f'{self.__class__.__name__}, '
                        f'training start from scratch')
            self.apply(_init_weights)
        else:
            self.apply(_init_weights)
            checkpoint = CheckpointLoader.load_checkpoint(
                checkpoint_path, logger=logger, map_location='cpu')
            state_dict = _extract_vitaev2_state_dict(checkpoint)
            incompatible = self.load_state_dict(state_dict, strict=False)
            logger.info(
                f'Loaded ViTAEv2 pretrained checkpoint from {checkpoint_path}. '
                f'missing_keys={len(incompatible.missing_keys)}, '
                f'unexpected_keys={len(incompatible.unexpected_keys)}'
            )

    def forwardTwoLayer(self, x):
        x1 = self.layers[0](x)
        x2 = self.layers[1](x1)
        return x1, x2
    
    def forwardThreeLayer(self, x):
        x0 = self.layers[1](x)
        x1 = self.layers[2](x0)
        x2 = self.layers[3](x1)
        return x0, x1, x2

    def forward_features(self, x):
        b, c, h, w = x.shape
        for layer in self.layers:
            x, (h, w) = layer(x, (h, w))
        return x

    # def forward(self, x):
    #     x = self.forward_features(x)
    #     return x

    def forward(self, x):
        """Forward function."""
        outs = []
        b, _, h, w = x.shape
        for i, layer in enumerate(self.layers):
            x, (h, w) = layer(x, (h, w))
            if i in self.out_indices:
                outs.append(x.reshape(b, h, w, -1).permute(0, 3, 1, 2).contiguous())

        return tuple(outs)

    def train(self, mode=True):
        """Convert the model into training mode while keep layers freezed."""
        super(ViTAEv2, self).train(mode)
        self._freeze_stages()
