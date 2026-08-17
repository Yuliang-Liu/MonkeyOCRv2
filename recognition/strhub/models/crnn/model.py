from typing import Sequence

import torch.nn as nn
from einops import rearrange

from strhub.models.modules import BidirectionalLSTM


class CRNN(nn.Module):

    def __init__(self, img_h, nc, nclass, nh, leaky_relu=False):
        super().__init__()
        assert img_h % 16 == 0, 'img_h has to be a multiple of 16'

        ks = [3, 3, 3, 3, 3, 3, 2]
        ps = [1, 1, 1, 1, 1, 1, 0]
        ss = [1, 1, 1, 1, 1, 1, 1]
        nm = [64, 128, 256, 256, 512, 512, 512]

        cnn = nn.Sequential()

        def convRelu(i, batchNormalization=False):
            nIn = nc if i == 0 else nm[i - 1]
            nOut = nm[i]
            cnn.add_module(f'conv{i}',
                           nn.Conv2d(nIn, nOut, ks[i], ss[i], ps[i], bias=not batchNormalization))
            if batchNormalization:
                cnn.add_module(f'batchnorm{i}', nn.BatchNorm2d(nOut))
            if leaky_relu:
                cnn.add_module(f'relu{i}',
                               nn.LeakyReLU(0.2, inplace=True))
            else:
                cnn.add_module(f'relu{i}', nn.ReLU(True))

        convRelu(0)
        cnn.add_module('pooling0', nn.MaxPool2d(2, 2))  # 64x16x64
        convRelu(1)
        cnn.add_module('pooling1', nn.MaxPool2d(2, 2))  # 128x8x32
        convRelu(2, True)
        convRelu(3)
        cnn.add_module('pooling2',
                       nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 256x4x16
        convRelu(4, True)
        convRelu(5)
        cnn.add_module('pooling3',
                       nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 512x2x16
        convRelu(6, True)  # 512x1x16

        self.cnn = cnn
        self.rnn = nn.Sequential(
            BidirectionalLSTM(512, nh, nh),
            BidirectionalLSTM(nh, nh, nclass))

    def forward(self, input):
        # conv features
        conv = self.cnn(input)
        b, c, h, w = conv.size()
        assert h == 1, 'the height of conv must be 1'
        conv = conv.squeeze(2)
        conv = conv.transpose(1, 2)  # [b, w, c]

        # rnn features
        output = self.rnn(conv)

        return output


class CRNNMonkey(nn.Module):

    def __init__(
        self,
        img_size: Sequence[int],
        patch_size: Sequence[int],
        nclass: int,
        nh: int,
        monkey_vit_dir: str,
    ):
        super().__init__()
        self.img_size = tuple(img_size)
        self.patch_size = tuple(patch_size)

        from transformers import AutoModel

        self.encoder = AutoModel.from_pretrained(
            str(monkey_vit_dir),
            trust_remote_code=True,
            dtype='auto',
        )
        embed_dim = int(getattr(self.encoder.config, 'embed_dim', 384))
        self.height_pool = nn.Linear(embed_dim, 1)
        self.rnn = nn.Sequential(
            BidirectionalLSTM(embed_dim, nh, nh),
            BidirectionalLSTM(nh, nh, nclass),
        )

    def no_weight_decay(self):
        return set()

    def _encode_width_sequence(self, x):
        if not isinstance(x, dict):
            raise TypeError('CRNNMonkey expects a batch dict with `pixel_values` and `image_grid_thw`.')

        pixel_values = x['pixel_values']
        image_grid_thw = x['image_grid_thw']
        param_dtype = next(self.encoder.parameters()).dtype
        pixel_values = pixel_values.to(dtype=param_dtype)
        vision_embeddings = self.encoder(pixel_values, image_grid_thw)

        spatial_merge_size = self.encoder.config.spatial_merge_size
        sequences = []
        lengths = []
        start = 0
        for _, grid_h, grid_w in image_grid_thw.tolist():
            num_tokens = grid_h * grid_w
            sample_embeddings = vision_embeddings[start : start + num_tokens]
            sample_feature = rearrange(
                sample_embeddings,
                '(h w m1 m2) c -> (h m1) (w m2) c',
                h=grid_h // spatial_merge_size,
                w=grid_w // spatial_merge_size,
                m1=spatial_merge_size,
                m2=spatial_merge_size,
            )
            height_logits = self.height_pool(sample_feature).squeeze(-1).transpose(0, 1)
            height_weights = height_logits.softmax(dim=-1).transpose(0, 1).unsqueeze(-1)
            width_sequence = (sample_feature * height_weights).sum(dim=0)
            sequences.append(width_sequence)
            lengths.append(width_sequence.shape[0])
            start += num_tokens

        max_length = max(lengths)
        batch_size = len(sequences)
        embed_dim = sequences[0].shape[-1]
        padded = vision_embeddings.new_zeros((batch_size, max_length, embed_dim))
        for idx, sequence in enumerate(sequences):
            padded[idx, : sequence.shape[0]] = sequence
        return padded

    def forward(self, input):
        sequence = self._encode_width_sequence(input)
        return self.rnn(sequence)
