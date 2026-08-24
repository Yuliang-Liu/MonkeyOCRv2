from torch import nn

from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from segmentation_models_pytorch.decoders.unet.decoder import UnetDecoder

from mmseg.registry import MODELS


@MODELS.register_module()
class ScseUnetDecoderHead(BaseDecodeHead):
    def __init__(self, n_blocks, decoder_channels, **kwargs):
        if kwargs.get('pool_scales') is not None:
            kwargs.pop('pool_scales')
        super().__init__(**kwargs)
        self.n_blocks = n_blocks
        self.decoder_channels = decoder_channels
        self.decoder = UnetDecoder(
            encoder_channels=[None, *self.in_channels],
            decoder_channels=self.decoder_channels,
            n_blocks=self.n_blocks,
            attention_type='scse',
            center=False,
        )

        # self.up_sample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, inputs):
        x = self._transform_inputs(inputs)
        x = self.decoder(None, *x)
        x = self.cls_seg(x)
        # x = self.up_sample(x)

        return x

