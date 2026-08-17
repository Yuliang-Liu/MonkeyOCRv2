from functools import partial
from typing import Optional, Sequence

import torch
import torch.nn as nn
from torch import Tensor

from timm.models.helpers import named_apply

from strhub.data.utils import Tokenizer
from strhub.models.utils import init_weights

from .modules import Decoder, DecoderLayer, Encoder, TokenEmbedding
from .monkey_encoder import MonkeyEncoder


class PARSeq(nn.Module):
    def __init__(
        self,
        num_tokens: int,
        max_label_length: int,
        img_size: Sequence[int],
        patch_size: Sequence[int],
        embed_dim: int,
        enc_num_heads: int,
        enc_mlp_ratio: int,
        enc_depth: int,
        dec_num_heads: int,
        dec_mlp_ratio: int,
        dec_depth: int,
        encoder_backend: str,
        monkey_vit_dir: Optional[str],
        decode_ar: bool,
        refine_iters: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.max_label_length = max_label_length
        self.decode_ar = decode_ar
        self.refine_iters = refine_iters

        if encoder_backend == 'baseline':
            self.encoder = Encoder(
                img_size,
                patch_size,
                embed_dim=embed_dim,
                depth=enc_depth,
                num_heads=enc_num_heads,
                mlp_ratio=enc_mlp_ratio,
            )
        elif encoder_backend == 'monkey':
            self.encoder = MonkeyEncoder(img_size, patch_size, monkey_vit_dir or None)
        else:
            raise ValueError(f'Unsupported encoder_backend: {encoder_backend}')
        decoder_layer = DecoderLayer(embed_dim, dec_num_heads, embed_dim * dec_mlp_ratio, dropout)
        self.decoder = Decoder(decoder_layer, num_layers=dec_depth, norm=nn.LayerNorm(embed_dim))
        self.head = nn.Linear(embed_dim, num_tokens - 2)
        self.text_embed = TokenEmbedding(num_tokens, embed_dim)
        self.pos_queries = nn.Parameter(torch.Tensor(1, max_label_length + 1, embed_dim))
        self.dropout = nn.Dropout(p=dropout)
        named_apply(partial(init_weights, exclude=['encoder']), self)
        nn.init.trunc_normal_(self.pos_queries, std=0.02)

    @property
    def _device(self) -> torch.device:
        return next(self.head.parameters(recurse=False)).device

    @torch.jit.ignore
    def no_weight_decay(self):
        param_names = {'text_embed.embedding.weight', 'pos_queries'}
        enc_param_names = {'encoder.' + n for n in self.encoder.no_weight_decay()}
        return param_names.union(enc_param_names)

    def encode(self, img: torch.Tensor):
        encoded = self.encoder(img)
        if isinstance(encoded, tuple):
            return encoded
        return encoded, None

    def decode(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        memory_key_padding_mask: Optional[Tensor] = None,
        tgt_mask: Optional[Tensor] = None,
        tgt_padding_mask: Optional[Tensor] = None,
        tgt_query: Optional[Tensor] = None,
        tgt_query_mask: Optional[Tensor] = None,
    ):
        N, L = tgt.shape
        null_ctx = self.text_embed(tgt[:, :1])
        tgt_emb = self.pos_queries[:, : L - 1] + self.text_embed(tgt[:, 1:])
        tgt_emb = self.dropout(torch.cat([null_ctx, tgt_emb], dim=1))
        if tgt_query is None:
            tgt_query = self.pos_queries[:, :L].expand(N, -1, -1)
        tgt_query = self.dropout(tgt_query)
        memory = memory.to(dtype=tgt_query.dtype)
        return self.decoder(
            tgt_query,
            tgt_emb,
            memory,
            memory_key_padding_mask=memory_key_padding_mask,
            query_mask=tgt_query_mask,
            content_mask=tgt_mask,
            content_key_padding_mask=tgt_padding_mask,
        )

    def forward(self, tokenizer: Tokenizer, images: Tensor, max_length: Optional[int] = None) -> Tensor:
        testing = max_length is None
        max_length = self.max_label_length if max_length is None else min(max_length, self.max_label_length)
        bs = images['image_grid_thw'].shape[0] if isinstance(images, dict) else images.shape[0]
        num_steps = max_length + 1
        memory, memory_key_padding_mask = self.encode(images)
        pos_queries = self.pos_queries[:, :num_steps].expand(bs, -1, -1)
        tgt_mask = query_mask = torch.triu(torch.ones((num_steps, num_steps), dtype=torch.bool, device=self._device), 1)

        if self.decode_ar:
            tgt_in = torch.full((bs, num_steps), tokenizer.pad_id, dtype=torch.long, device=self._device)
            tgt_in[:, 0] = tokenizer.bos_id
            logits = []
            for i in range(num_steps):
                j = i + 1
                tgt_out = self.decode(
                    tgt_in[:, :j],
                    memory,
                    memory_key_padding_mask,
                    tgt_mask[:j, :j],
                    tgt_query=pos_queries[:, i:j],
                    tgt_query_mask=query_mask[i:j, :j],
                )
                p_i = self.head(tgt_out)
                logits.append(p_i)
                if j < num_steps:
                    tgt_in[:, j] = p_i.squeeze().argmax(-1)
                    if testing and (tgt_in == tokenizer.eos_id).any(dim=-1).all():
                        break
            logits = torch.cat(logits, dim=1)
        else:
            tgt_in = torch.full((bs, 1), tokenizer.bos_id, dtype=torch.long, device=self._device)
            tgt_out = self.decode(tgt_in, memory, memory_key_padding_mask, tgt_query=pos_queries)
            logits = self.head(tgt_out)

        if self.refine_iters:
            query_mask[torch.triu(torch.ones(num_steps, num_steps, dtype=torch.bool, device=self._device), 2)] = 0
            bos = torch.full((bs, 1), tokenizer.bos_id, dtype=torch.long, device=self._device)
            for _ in range(self.refine_iters):
                tgt_in = torch.cat([bos, logits[:, :-1].argmax(-1)], dim=1)
                tgt_padding_mask = (tgt_in == tokenizer.eos_id).int().cumsum(-1) > 0
                tgt_out = self.decode(
                    tgt_in,
                    memory,
                    memory_key_padding_mask,
                    tgt_mask,
                    tgt_padding_mask,
                    pos_queries,
                    query_mask[:, : tgt_in.shape[1]],
                )
                logits = self.head(tgt_out)

        return logits
