# Copyright (c) OpenMMLab. All rights reserved.

import timm
from mmengine.model import BaseModule
from mmengine.registry import MODELS as MMENGINE_MODELS
from mmseg.registry import MODELS


@MODELS.register_module()
class TimmDct(BaseModule):
    def __init__(
            self,
            model_name,
            pretrained=True,
            checkpoint_path='',
            in_channels=3,
            init_cfg=None,
            fusion='ZERO',
            **kwargs,
    ):
        super().__init__(init_cfg)
        if 'norm_layer' in kwargs:
            kwargs['norm_layer'] = MMENGINE_MODELS.get(kwargs['norm_layer'])
        if 'out_indices' in kwargs:
            self.out_indices = kwargs.pop('out_indices')
        assert model_name.startswith('convnextv2'), f'Unsupported model_name: {model_name}'
        self.model_name = model_name
        self.timm_model = timm.create_model(
            model_name=model_name,
            features_only=False,
            pretrained=pretrained,
            in_chans=in_channels,
            checkpoint_path=checkpoint_path,
            **kwargs,
        )
        # Make unused parameters None
        self.timm_model.head = None

        # Hack to use pretrained weights from timm
        if pretrained or checkpoint_path:
            self._is_init = True

        self.fph = FPH()
        self.fusion = fusion
        if fusion == 'SCSE':
            raise NotImplementedError
        elif fusion == 'ADD':
            raise NotImplementedError
        elif fusion == 'ZERO':
            self.FU = nn.ModuleList([nn.Sequential(SCSEModule(512), nn.Conv2d(512, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(True))])
            self.FU.append(nn.Conv2d(256, 256, 1, 1, 0))
            self.FU[1].weight.data.zero_()

        else:
            raise NotImplementedError

    def forward(self, x):
        x, dct, qtb = x['x'], x['dct'], x['qtb']
        f_dct = self.fph(dct, qtb)
        outs = []

        if self.model_name.startswith('convnextv2'):
            x = self.timm_model.stem(x)
            for i, layer in enumerate(self.timm_model.stages):
                x = layer(x)
                if self.fusion in ['SCSE', 'ADD'] and i == 1:
                    raise NotImplementedError
                elif self.fusion == 'ZERO' and i == 1:
                    ext = self.FU[0](torch.cat((x, f_dct), dim=1))
                    x = self.FU[1](ext) + x
                if i in self.out_indices:
                    outs.append(x)
            return outs

        raise NotImplementedError


from efficientnet_pytorch.utils import *
import torch
import torch.nn as nn
import torch._utils
import torch.nn.functional as F
import collections

BlockArgs = collections.namedtuple('BlockArgs', ['num_repeat', 'kernel_size', 'stride', 'expand_ratio', 'input_filters',
                                                 'output_filters', 'se_ratio', 'id_skip'])
GlobalParams = collections.namedtuple('GlobalParams',
                                      ['width_coefficient', 'depth_coefficient', 'image_size', 'dropout_rate',
                                       'num_classes', 'batch_norm_momentum', 'batch_norm_epsilon', 'drop_connect_rate',
                                       'depth_divisor', 'min_depth', 'include_top'])
global_params = GlobalParams(width_coefficient=1.8, depth_coefficient=2.6, image_size=528, dropout_rate=0.0,
                             num_classes=1000, batch_norm_momentum=0.99, batch_norm_epsilon=0.001,
                             drop_connect_rate=0.0, depth_divisor=8, min_depth=None, include_top=True)

def get_width_and_height_from_size(x):
    if isinstance(x, int):
        return x, x
    if isinstance(x, list) or isinstance(x, tuple):
        return x
    else:
        raise TypeError()


def calculate_output_image_size(input_image_size, stride):
    if input_image_size is None:
        return None
    image_height, image_width = get_width_and_height_from_size(input_image_size)
    stride = stride if isinstance(stride, int) else stride[0]
    image_height = int(math.ceil(image_height / stride))
    image_width = int(math.ceil(image_width / stride))
    return [image_height, image_width]


class MBConvBlock(nn.Module):
    def __init__(self, block_args, global_params, image_size=25):
        super().__init__()
        self._block_args = block_args
        self._bn_mom = 1 - global_params.batch_norm_momentum  # pytorch's difference from tensorflow
        self._bn_eps = global_params.batch_norm_epsilon
        self.has_se = (self._block_args.se_ratio is not None) and (0 < self._block_args.se_ratio <= 1)
        self.id_skip = block_args.id_skip  # whether to use skip connection and drop connect
        inp = self._block_args.input_filters  # number of input channels
        oup = self._block_args.input_filters * self._block_args.expand_ratio  # number of output channels
        if self._block_args.expand_ratio != 1:
            Conv2d = get_same_padding_conv2d(image_size=image_size)
            self._expand_conv = Conv2d(in_channels=inp, out_channels=oup, kernel_size=1, bias=False)
            self._bn0 = nn.BatchNorm2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)
        k = self._block_args.kernel_size
        s = self._block_args.stride
        Conv2d = get_same_padding_conv2d(image_size=image_size)
        self._depthwise_conv = Conv2d(
            in_channels=oup, out_channels=oup, groups=oup,  # groups makes it depthwise
            kernel_size=k, stride=s, bias=False)
        self._bn1 = nn.BatchNorm2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)
        image_size = calculate_output_image_size(image_size, s)
        if self.has_se:
            Conv2d = get_same_padding_conv2d(image_size=(1, 1))
            num_squeezed_channels = max(1, int(self._block_args.input_filters * self._block_args.se_ratio))
            self._se_reduce = Conv2d(in_channels=oup, out_channels=num_squeezed_channels, kernel_size=1)
            self._se_expand = Conv2d(in_channels=num_squeezed_channels, out_channels=oup, kernel_size=1)
        final_oup = self._block_args.output_filters
        Conv2d = get_same_padding_conv2d(image_size=image_size)
        self._project_conv = Conv2d(in_channels=oup, out_channels=final_oup, kernel_size=1, bias=False)
        self._bn2 = nn.BatchNorm2d(num_features=final_oup, momentum=self._bn_mom, eps=self._bn_eps)
        self._swish = MemoryEfficientSwish()

    def forward(self, inputs, drop_connect_rate=None):
        x = inputs
        if self._block_args.expand_ratio != 1:
            x = self._expand_conv(inputs)
            x = self._bn0(x)
            x = self._swish(x)
        x = self._depthwise_conv(x)
        x = self._bn1(x)
        x = self._swish(x)
        if self.has_se:
            x_squeezed = F.adaptive_avg_pool2d(x, 1)
            x_squeezed = self._se_reduce(x_squeezed)
            x_squeezed = self._swish(x_squeezed)
            x_squeezed = self._se_expand(x_squeezed)
            x = torch.sigmoid(x_squeezed) * x
        x = self._project_conv(x)
        x = self._bn2(x)
        input_filters, output_filters = self._block_args.input_filters, self._block_args.output_filters
        if self.id_skip and self._block_args.stride == 1 and input_filters == output_filters:
            if drop_connect_rate:
                x = drop_connect(x, p=drop_connect_rate, training=self.training)
            x = x + inputs  # skip connection
        return x

    def set_swish(self, memory_efficient=True):
        self._swish = MemoryEfficientSwish() if memory_efficient else Swish()


class AddCoords(nn.Module):
    def __init__(self, with_r=True):
        super().__init__()
        self.with_r = with_r

    def forward(self, input_tensor):
        batch_size, _, x_dim, y_dim = input_tensor.size()
        xx_c, yy_c = torch.meshgrid(torch.arange(x_dim, dtype=input_tensor.dtype, device=input_tensor.device),
                                    torch.arange(y_dim, dtype=input_tensor.dtype, device=input_tensor.device))
        xx_c = xx_c.to(input_tensor.device) / (x_dim - 1) * 2 - 1
        yy_c = yy_c.to(input_tensor.device) / (y_dim - 1) * 2 - 1
        xx_c = xx_c.expand(batch_size, 1, x_dim, y_dim)
        yy_c = yy_c.expand(batch_size, 1, x_dim, y_dim)
        ret = torch.cat((input_tensor, xx_c, yy_c), dim=1)
        if self.with_r:
            rr = torch.sqrt(torch.pow(xx_c - 0.5, 2) + torch.pow(yy_c - 0.5, 2))
            ret = torch.cat([ret, rr], dim=1)
        return ret


class FPH(nn.Module):

    def __init__(self):
        super(FPH, self).__init__()
        self.obembed = nn.Embedding(21, 21).from_pretrained(torch.eye(21))
        self.qtembed = nn.Embedding(64, 16)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=21, out_channels=64, kernel_size=3, stride=1, dilation=8, padding=8),
            nn.BatchNorm2d(64, momentum=0.01), nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=16, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(16, momentum=0.01), nn.ReLU(inplace=True))
        self.addcoords = AddCoords()
        repeats = (1, 1, 1)
        in_channles = (256, 256, 256)
        out_channles = (256, 256, 512)
        self.conv0 = nn.Sequential(
            nn.Conv2d(in_channels=35, out_channels=256, kernel_size=8, stride=8, padding=0, bias=False),
            nn.BatchNorm2d(256, momentum=0.01),
            nn.ReLU(inplace=True),
            MBConvBlock(BlockArgs(num_repeat=repeats[0], kernel_size=3, stride=[1], expand_ratio=6,
                                  input_filters=in_channles[0], output_filters=in_channles[1], se_ratio=0.25,
                                  id_skip=True), global_params),
            MBConvBlock(BlockArgs(num_repeat=repeats[0], kernel_size=3, stride=[1], expand_ratio=6,
                                  input_filters=in_channles[1], output_filters=in_channles[1], se_ratio=0.25,
                                  id_skip=True), global_params),
            MBConvBlock(BlockArgs(num_repeat=repeats[0], kernel_size=3, stride=[1], expand_ratio=6,
                                  input_filters=in_channles[1], output_filters=in_channles[1], se_ratio=0.25,
                                  id_skip=True), global_params), )

    def forward(self, x, qtable):
        x = self.conv2(self.conv1(self.obembed(x).permute(0, 3, 1, 2).contiguous()))
        B, C, H, W = x.shape
        return self.conv0(self.addcoords(torch.cat(((x.reshape(B, C, H // 8, 8, W // 8, 8).permute(0, 1, 3, 5, 2,
                                                                                                   4) * self.qtembed(
            qtable.unsqueeze(-1).unsqueeze(-1).long()).transpose(1, 6).squeeze(6).contiguous()).permute(0, 1, 4, 2, 5,
                                                                                                        3).reshape(B, C,
                                                                                                                   H,
                                                                                                                   W),
                                                    x), dim=1)))


class SCSEModule(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.cSE = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1),
            nn.Sigmoid(),
        )
        self.sSE = nn.Sequential(nn.Conv2d(in_channels, 1, 1), nn.Sigmoid())

    def forward(self, x):
        return x * self.cSE(x) + x * self.sSE(x)
