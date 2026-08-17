from pathlib import Path

import torch
from einops import rearrange
from torch import nn


DEFAULT_MONKEY_VIT_DIR = Path(__file__).resolve().parents[3] / 'pretrained' / 'monkeyocr_vit'


class MonkeyEncoder(nn.Module):
    def __init__(self, img_size, patch_size, monkey_vit_dir: str = str(DEFAULT_MONKEY_VIT_DIR)):
        super().__init__()
        self.img_size = tuple(img_size)
        self.patch_size = tuple(patch_size)
        self.monkey_vit_dir = Path(monkey_vit_dir)

        from transformers import AutoModel

        self.backbone = AutoModel.from_pretrained(
            str(self.monkey_vit_dir),
            trust_remote_code=True,
            dtype='auto',
        )

    def no_weight_decay(self):
        return set()

    def forward(self, x):
        if not isinstance(x, dict):
            raise TypeError('MonkeyEncoder expects a batch dict with `pixel_values` and `image_grid_thw`.')

        pixel_values = x['pixel_values']
        image_grid_thw = x['image_grid_thw']
        param_dtype = next(self.backbone.parameters()).dtype
        pixel_values = pixel_values.to(dtype=param_dtype)
        vision_embeddings = self.backbone(pixel_values, image_grid_thw)

        spatial_merge_size = self.backbone.config.spatial_merge_size
        outputs = []
        lengths = []
        start = 0
        for _, grid_h, grid_w in image_grid_thw.tolist():
            num_tokens = grid_h * grid_w
            sample_embeddings = vision_embeddings[start:start + num_tokens]
            sample_feature = rearrange(
                sample_embeddings,
                '(h w m1 m2) c -> (h m1) (w m2) c',
                h=grid_h // spatial_merge_size,
                w=grid_w // spatial_merge_size,
                m1=spatial_merge_size,
                m2=spatial_merge_size,
            )
            tokens = sample_feature.reshape(num_tokens, -1)
            outputs.append(tokens)
            lengths.append(tokens.shape[0])
            start += num_tokens

        max_tokens = max(lengths)
        batch_size = len(outputs)
        embed_dim = outputs[0].shape[-1]
        padded = vision_embeddings.new_zeros((batch_size, max_tokens, embed_dim))
        memory_key_padding_mask = torch.ones((batch_size, max_tokens), dtype=torch.bool, device=vision_embeddings.device)
        for idx, sample_tokens in enumerate(outputs):
            sample_len = sample_tokens.shape[0]
            padded[idx, :sample_len] = sample_tokens
            memory_key_padding_mask[idx, :sample_len] = False

        return padded, memory_key_padding_mask
