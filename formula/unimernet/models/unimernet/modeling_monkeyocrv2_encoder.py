import math
import os
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from safetensors.torch import load_file as safetensors_load_file
from transformers.modeling_outputs import BaseModelOutput
from transformers.modeling_utils import PreTrainedModel

from .configuration_monkeyocrv2_encoder import MonkeyOCRv2VisionConfig


flash_attn_available = True
npu_available = True

try:
    from flash_attn import flash_attn_varlen_func
except ImportError:
    flash_attn_available = False

try:
    import torch_npu
except ImportError:
    npu_available = False


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb_vision(tensor: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    orig_dtype = tensor.dtype
    tensor = tensor.float()

    cos = freqs.cos()
    sin = freqs.sin()

    cos = cos.unsqueeze(1).repeat(1, 1, 2).unsqueeze(0).float()
    sin = sin.unsqueeze(1).repeat(1, 1, 2).unsqueeze(0).float()

    output = (tensor * cos) + (rotate_half(tensor) * sin)
    return output.to(orig_dtype)


def flatten_grid_to_token_sequence(
    hidden_states: torch.Tensor,
    merge_size: int,
    token_order: str,
) -> torch.Tensor:
    if token_order == "row_major" or merge_size == 1:
        return hidden_states.flatten(2).transpose(1, 2).contiguous()
    if token_order != "native_merge":
        raise ValueError(f"Unsupported token order: {token_order}")

    batch_size, channels, grid_h, grid_w = hidden_states.shape
    hidden_states = hidden_states.view(
        batch_size,
        channels,
        grid_h // merge_size,
        merge_size,
        grid_w // merge_size,
        merge_size,
    )
    hidden_states = hidden_states.permute(0, 2, 4, 3, 5, 1).contiguous()
    return hidden_states.view(batch_size, grid_h * grid_w, channels)


def token_sequence_to_grid(
    hidden_states: torch.Tensor,
    grid_size: Tuple[int, int],
    merge_size: int,
    token_order: str,
) -> torch.Tensor:
    batch_size, seq_length, channels = hidden_states.shape
    grid_h, grid_w = grid_size
    if seq_length != grid_h * grid_w:
        raise ValueError(
            f"Sequence length {seq_length} does not match grid size {grid_size}."
        )

    if token_order == "row_major" or merge_size == 1:
        return hidden_states.transpose(1, 2).contiguous().view(batch_size, channels, grid_h, grid_w)
    if token_order != "native_merge":
        raise ValueError(f"Unsupported token order: {token_order}")

    hidden_states = hidden_states.view(
        batch_size,
        grid_h // merge_size,
        grid_w // merge_size,
        merge_size,
        merge_size,
        channels,
    )
    hidden_states = hidden_states.permute(0, 5, 1, 3, 2, 4).contiguous()
    return hidden_states.view(batch_size, channels, grid_h, grid_w)


@dataclass
class MonkeyOCRv2EncoderOutput(BaseModelOutput):
    grid_size: Optional[Tuple[int, int]] = None
    token_order: Optional[str] = None
    image_grid_thw: Optional[torch.Tensor] = None
    attention_mask: Optional[torch.Tensor] = None


class VisionRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seqlen: int) -> torch.Tensor:
        seq = torch.arange(seqlen, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        return torch.outer(seq, self.inv_freq)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._norm(x.float()).type_as(x) * self.weight


class SwiGLUFFN(nn.Module):
    def __init__(self, config: MonkeyOCRv2VisionConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.embed_dim, config.intermediate_size, bias=config.use_bias)
        self.fc2 = nn.Linear(config.intermediate_size, config.embed_dim, bias=config.use_bias)
        self.fc3 = nn.Linear(config.embed_dim, config.intermediate_size, bias=config.use_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.silu(self.fc1(x)) * self.fc3(x))


class PatchEmbed(nn.Module):
    def __init__(self, config: MonkeyOCRv2VisionConfig):
        super().__init__()
        self.patch_size = config.patch_size
        self.embed_dim = config.embed_dim
        self.merge_size = config.spatial_merge_size
        self.token_order = "native_merge" if config.use_native_token_order else "row_major"
        self.proj = nn.Conv2d(
            config.num_channels,
            config.embed_dim,
            kernel_size=(config.patch_size, config.patch_size),
            stride=(config.patch_size, config.patch_size),
        )
        self.norm = RMSNorm(config.embed_dim, eps=config.rms_norm_eps)

    def forward(self, pixel_values: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        hidden_states = self.proj(pixel_values)
        _, _, grid_h, grid_w = hidden_states.shape
        hidden_states = flatten_grid_to_token_sequence(
            hidden_states,
            merge_size=self.merge_size,
            token_order=self.token_order,
        )
        hidden_states = self.norm(hidden_states)
        return hidden_states, (grid_h, grid_w)


class ViTPreprocessor(nn.Module):
    def __init__(self, config: MonkeyOCRv2VisionConfig):
        super().__init__()
        self.patchifier = PatchEmbed(config)

    def forward(self, pixel_values: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        return self.patchifier(pixel_values)


class VisionAttention(nn.Module):
    def __init__(self, config: MonkeyOCRv2VisionConfig, dim: int, num_heads: int, bias: bool) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=bias)
        self.proj = nn.Linear(dim, dim, bias=bias)

    def forward(self, hidden_states: torch.Tensor, cu_seqlens: torch.Tensor, rotary_pos_emb: torch.Tensor) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        q, k, v = self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        q = apply_rotary_pos_emb_vision(q.unsqueeze(0), rotary_pos_emb).squeeze(0)
        k = apply_rotary_pos_emb_vision(k.unsqueeze(0), rotary_pos_emb).squeeze(0)

        attention_mask = torch.full(
            [1, seq_length, seq_length],
            torch.finfo(q.dtype).min,
            device=q.device,
            dtype=q.dtype,
        )
        for i in range(1, len(cu_seqlens)):
            attention_mask[..., cu_seqlens[i - 1] : cu_seqlens[i], cu_seqlens[i - 1] : cu_seqlens[i]] = 0

        q = q.transpose(0, 1)
        k = k.transpose(0, 1)
        v = v.transpose(0, 1)
        attn_weights = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(self.head_dim)
        attn_weights = nn.functional.softmax(attn_weights + attention_mask, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(0, 1).reshape(seq_length, -1)
        return self.proj(attn_output)


class VisionAttentionV2(nn.Module):
    def __init__(self, config: MonkeyOCRv2VisionConfig, dim: int, num_heads: int, bias: bool) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=bias)
        self.proj = nn.Linear(dim, dim, bias=bias)

    def forward(self, hidden_states: torch.Tensor, cu_seqlens: torch.Tensor, rotary_pos_emb: torch.Tensor) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        q, k, v = self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        q = apply_rotary_pos_emb_vision(q.unsqueeze(0), rotary_pos_emb).squeeze(0)
        k = apply_rotary_pos_emb_vision(k.unsqueeze(0), rotary_pos_emb).squeeze(0)

        seqlens = torch.diff(cu_seqlens).tolist()
        q_list = torch.split(q, seqlens, 0)
        k_list = torch.split(k, seqlens, 0)
        v_list = torch.split(v, seqlens, 0)

        outputs = []
        for q_i, k_i, v_i in zip(q_list, k_list, v_list):
            q_i = q_i.transpose(0, 1)
            k_i = k_i.transpose(0, 1)
            v_i = v_i.transpose(0, 1)
            out = torch.matmul(q_i, k_i.transpose(1, 2)) / math.sqrt(self.head_dim)
            out = nn.functional.softmax(out, dim=-1, dtype=torch.float32).to(q.dtype)
            outputs.append(torch.matmul(out, v_i).transpose(0, 1))

        attn_output = torch.concat(outputs, dim=0).reshape(seq_length, -1)
        return self.proj(attn_output)


class VisionFlashAttention2(nn.Module):
    def __init__(self, config: MonkeyOCRv2VisionConfig, dim: int, num_heads: int, bias: bool) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=bias)
        self.proj = nn.Linear(dim, dim, bias=bias)
        self.is_causal = config.is_causal

    def forward(self, hidden_states: torch.Tensor, cu_seqlens: torch.Tensor, rotary_pos_emb: torch.Tensor) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        q, k, v = self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        q = apply_rotary_pos_emb_vision(q.unsqueeze(0), rotary_pos_emb).squeeze(0)
        k = apply_rotary_pos_emb_vision(k.unsqueeze(0), rotary_pos_emb).squeeze(0)
        max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
        attn_output = flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens,
            cu_seqlens,
            max_seqlen,
            max_seqlen,
            causal=self.is_causal,
        ).reshape(seq_length, -1)
        return self.proj(attn_output)


class VisionAscendAttention(nn.Module):
    def __init__(self, config: MonkeyOCRv2VisionConfig, dim: int, num_heads: int, bias: bool) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=bias)
        self.proj = nn.Linear(dim, dim, bias=bias)

    def forward(self, hidden_states: torch.Tensor, cu_seqlens: torch.Tensor, rotary_pos_emb: torch.Tensor) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        q, k, v = self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        q = apply_rotary_pos_emb_vision(q.unsqueeze(0), rotary_pos_emb).squeeze(0)
        k = apply_rotary_pos_emb_vision(k.unsqueeze(0), rotary_pos_emb).squeeze(0)

        attention_mask = torch.ones([1, seq_length, seq_length], device=q.device, dtype=torch.bool)
        for i in range(1, len(cu_seqlens)):
            attention_mask[..., cu_seqlens[i - 1] : cu_seqlens[i], cu_seqlens[i - 1] : cu_seqlens[i]] = False

        q = q.transpose(0, 1).unsqueeze(0)
        k = k.transpose(0, 1).unsqueeze(0)
        v = v.transpose(0, 1).unsqueeze(0)

        attn_output = torch_npu.npu_prompt_flash_attention(
            q,
            k,
            v,
            atten_mask=attention_mask,
            num_heads=self.num_heads,
            input_layout="BNSD",
            scale_value=self.head_dim ** -0.5,
        )
        attn_output = attn_output.squeeze(0).transpose(0, 1).reshape(seq_length, -1)
        return self.proj(attn_output)


class VisionSdpaAttention(nn.Module):
    def __init__(self, config: MonkeyOCRv2VisionConfig, dim: int, num_heads: int, bias: bool) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=bias)
        self.proj = nn.Linear(dim, dim, bias=bias)

    def forward(self, hidden_states: torch.Tensor, cu_seqlens: torch.Tensor, rotary_pos_emb: torch.Tensor) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        q, k, v = self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        q = apply_rotary_pos_emb_vision(q.unsqueeze(0), rotary_pos_emb).squeeze(0)
        k = apply_rotary_pos_emb_vision(k.unsqueeze(0), rotary_pos_emb).squeeze(0)

        attention_mask = torch.zeros([1, seq_length, seq_length], device=q.device, dtype=torch.bool)
        for i in range(1, len(cu_seqlens)):
            attention_mask[..., cu_seqlens[i - 1] : cu_seqlens[i], cu_seqlens[i - 1] : cu_seqlens[i]] = True

        q = q.transpose(0, 1).unsqueeze(0)
        k = k.transpose(0, 1).unsqueeze(0)
        v = v.transpose(0, 1).unsqueeze(0)

        if attention_mask.stride(-1) != 1:
            attention_mask = torch.empty_like(attention_mask, memory_format=torch.contiguous_format).copy_(attention_mask)

        attn_output = F.scaled_dot_product_attention(q, k, v, attention_mask, dropout_p=0.0)

        attn_output = attn_output.squeeze(0).transpose(0, 1).reshape(seq_length, -1)
        return self.proj(attn_output)


VISION_ATTENTION_CLASSES = {
    "eager": VisionAttention,
    "eager_v2": VisionAttentionV2,
    "flash_attention_2": VisionFlashAttention2,
    "sdpa": VisionSdpaAttention,
    "ascend_fa": VisionAscendAttention,
}


class VisionBlock(nn.Module):
    def __init__(self, config: MonkeyOCRv2VisionConfig, attn_implementation: str):
        super().__init__()
        if attn_implementation == "flash_attention_2" and not flash_attn_available:
            attn_implementation = "ascend_fa" if npu_available else "eager_v2"
        if attn_implementation == "ascend_fa" and not npu_available:
            attn_implementation = "eager_v2"

        self.attn = VISION_ATTENTION_CLASSES[attn_implementation](
            config,
            config.embed_dim,
            num_heads=config.num_attention_heads,
            bias=config.use_bias,
        )
        self.norm1 = RMSNorm(config.embed_dim, eps=config.rms_norm_eps)
        self.mlp = SwiGLUFFN(config)
        self.norm2 = RMSNorm(config.embed_dim, eps=config.rms_norm_eps)

    def forward(self, hidden_states: torch.Tensor, cu_seqlens: torch.Tensor, rotary_pos_emb: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states),
            cu_seqlens=cu_seqlens,
            rotary_pos_emb=rotary_pos_emb,
        )
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states


class MonkeyOCRv2VisionTransformer(PreTrainedModel):
    config_class = MonkeyOCRv2VisionConfig
    main_input_name = "pixel_values"
    _supports_flash_attn = True
    _supports_sdpa = True
    _no_split_modules = ["VisionBlock"]

    def __init__(self, config: MonkeyOCRv2VisionConfig) -> None:
        super().__init__(config)
        self.config.hidden_size = config.embed_dim
        self.spatial_merge_size = config.spatial_merge_size
        self.token_order = "native_merge" if config.use_native_token_order else "row_major"

        self.patch_embed = ViTPreprocessor(config)
        self._init_weights(self.patch_embed.patchifier.proj)

        head_dim = config.embed_dim // config.num_attention_heads
        self.rotary_pos_emb = VisionRotaryEmbedding(head_dim // 2)
        self.blocks = nn.ModuleList(
            [VisionBlock(config, config.vision_attn_implementation) for _ in range(config.num_hidden_layers)]
        )

        if config.post_norm:
            self.post_trunk_norm = RMSNorm(config.embed_dim, eps=config.rms_norm_eps)

        self.gradient_checkpointing = config.gradient_checkpointing
        self._gradient_checkpointing_func = torch.utils.checkpoint.checkpoint

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, (nn.Linear, nn.Conv2d, nn.Conv3d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def get_pos_ids_by_grid(self, grid_thw: torch.Tensor) -> list[torch.Tensor]:
        pos_ids = []
        for t, h, w in grid_thw:
            hpos_ids = torch.arange(h, device=grid_thw.device).unsqueeze(1).expand(-1, w)
            hpos_ids = hpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            hpos_ids = hpos_ids.permute(0, 2, 1, 3).flatten()

            wpos_ids = torch.arange(w, device=grid_thw.device).unsqueeze(0).expand(h, -1)
            wpos_ids = wpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            wpos_ids = wpos_ids.permute(0, 2, 1, 3).flatten()

            pos_ids.append(torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1))

        return pos_ids

    def rot_pos_emb(self, grid_thw: torch.Tensor) -> torch.Tensor:
        pos_ids = torch.cat(self.get_pos_ids_by_grid(grid_thw), dim=0)
        max_grid_size = int(grid_thw[:, 1:].max().item())
        rotary_pos_emb_full = self.rotary_pos_emb(max_grid_size)
        emb = rotary_pos_emb_full[pos_ids]
        return torch.stack([emb[:, 0], emb[:, 1]], dim=2).reshape(emb.shape[0], -1)

    def _build_grid_thw(self, batch_size: int, grid_h: int, grid_w: int, device: torch.device) -> torch.Tensor:
        return torch.tensor([[1, grid_h, grid_w]], device=device, dtype=torch.long).repeat(batch_size, 1)

    def _embed_packed_pixel_values(self, pixel_values: torch.Tensor) -> torch.Tensor:
        patch_size = self.config.patch_size
        temporal_patch_size = self.config.temporal_patch_size
        num_channels = self.config.num_channels
        patch_dim = num_channels * temporal_patch_size * patch_size * patch_size
        if pixel_values.shape[-1] != patch_dim:
            raise ValueError(f"Expected packed Monkey patches with dim {patch_dim}, got {tuple(pixel_values.shape)}.")

        patchifier = self.patch_embed.patchifier
        hidden_states = pixel_values.view(
            -1,
            num_channels,
            temporal_patch_size,
            patch_size,
            patch_size,
        )[:, :, 0]
        hidden_states = patchifier.proj(hidden_states).view(-1, self.config.embed_dim)
        return patchifier.norm(hidden_states)

    @staticmethod
    def _pad_flat_hidden_states(
        hidden_states: torch.Tensor,
        grid_thw: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        lengths = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).tolist()
        chunks = torch.split(hidden_states, lengths, dim=0)
        batch_size = len(chunks)
        max_length = max(chunk.shape[0] for chunk in chunks)
        hidden_dim = hidden_states.shape[-1]
        padded = hidden_states.new_zeros((batch_size, max_length, hidden_dim))
        attention_mask = torch.zeros((batch_size, max_length), dtype=torch.long, device=hidden_states.device)
        for index, chunk in enumerate(chunks):
            padded[index, : chunk.shape[0]] = chunk
            attention_mask[index, : chunk.shape[0]] = 1
        return padded, attention_mask

    def _validate_input_resolution(self, height: int, width: int) -> None:
        if height % self.config.patch_size != 0 or width % self.config.patch_size != 0:
            raise ValueError(
                "MonkeyOCRv2 encoder requires height and width divisible by patch_size "
                f"{self.config.patch_size}, got {(height, width)}."
            )
        grid_h = height // self.config.patch_size
        grid_w = width // self.config.patch_size
        if grid_h % self.spatial_merge_size != 0 or grid_w % self.spatial_merge_size != 0:
            raise ValueError(
                "MonkeyOCRv2 encoder requires token grid divisible by spatial_merge_size "
                f"{self.spatial_merge_size}, got {(grid_h, grid_w)}."
            )

    def load_weights(self, model_path: str) -> None:
        if os.path.isdir(model_path):
            safetensors_path = os.path.join(model_path, "model.safetensors")
            bin_path = os.path.join(model_path, "pytorch_model.bin")
        else:
            safetensors_path = model_path if model_path.endswith(".safetensors") else ""
            bin_path = model_path if model_path.endswith(".bin") else ""

        if safetensors_path and os.path.isfile(safetensors_path):
            state_dict = safetensors_load_file(safetensors_path)
        elif bin_path and os.path.isfile(bin_path):
            state_dict = torch.load(bin_path, map_location="cpu")
        else:
            raise FileNotFoundError(f"Cannot find MonkeyOCRv2 encoder weights under '{model_path}'.")

        self.load_state_dict(state_dict, strict=True)

    def forward(
        self,
        pixel_values: torch.Tensor,
        image_grid_thw: Optional[torch.Tensor] = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        **kwargs,
    ) -> BaseModelOutput | tuple[torch.Tensor]:
        del output_attentions
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states

        packed_native_input = pixel_values.ndim == 2
        if packed_native_input:
            if image_grid_thw is None:
                raise ValueError("Packed Monkey pixel_values require image_grid_thw.")
            grid_thw = image_grid_thw.to(device=pixel_values.device, dtype=torch.long)
            hidden_states = self._embed_packed_pixel_values(pixel_values)
            batch_size = int(grid_thw[:, 0].sum().item())
            grid_h = grid_w = None
        else:
            if pixel_values.ndim != 4:
                raise ValueError(f"Expected BCHW or packed native pixel_values, got shape {tuple(pixel_values.shape)}.")

            batch_size, _, height, width = pixel_values.shape
            self._validate_input_resolution(height, width)

            hidden_states, (grid_h, grid_w) = self.patch_embed(pixel_values)
            grid_thw = self._build_grid_thw(batch_size, grid_h, grid_w, pixel_values.device)
        rotary_pos_emb = self.rot_pos_emb(grid_thw)
        cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
            dim=0,
            dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
        )
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

        hidden_dim = hidden_states.shape[-1]
        all_hidden_states = () if output_hidden_states else None

        for blk in self.blocks:
            if output_hidden_states:
                if packed_native_input:
                    all_hidden_states += (self._pad_flat_hidden_states(hidden_states, grid_thw)[0],)
                else:
                    all_hidden_states += (hidden_states.reshape(batch_size, -1, hidden_dim),)
            hidden_states = hidden_states.reshape(-1, hidden_dim)
            if self.gradient_checkpointing and self.training:
                hidden_states = self._gradient_checkpointing_func(
                    blk.__call__,
                    hidden_states,
                    cu_seqlens,
                    rotary_pos_emb,
                )
            else:
                hidden_states = blk(hidden_states, cu_seqlens=cu_seqlens, rotary_pos_emb=rotary_pos_emb)

        if self.config.post_norm:
            hidden_states = self.post_trunk_norm(hidden_states)

        attention_mask = None
        if packed_native_input:
            last_hidden_state, attention_mask = self._pad_flat_hidden_states(hidden_states, grid_thw)
        else:
            last_hidden_state = hidden_states.reshape(batch_size, -1, hidden_dim)
        if output_hidden_states:
            all_hidden_states += (last_hidden_state,)

        if not return_dict:
            return (last_hidden_state, all_hidden_states)

        return MonkeyOCRv2EncoderOutput(
            last_hidden_state=last_hidden_state,
            hidden_states=all_hidden_states,
            attentions=None,
            grid_size=(grid_h, grid_w),
            token_order=self.token_order,
            image_grid_thw=grid_thw,
            attention_mask=attention_mask,
        )
