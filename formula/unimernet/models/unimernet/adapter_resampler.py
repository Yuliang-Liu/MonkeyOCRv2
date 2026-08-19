from typing import Tuple

import torch
import torch.nn as nn

from .modeling_monkeyocrv2_encoder import token_sequence_to_grid


def _resolve_group_count(num_channels: int) -> int:
    for group_count in (32, 24, 16, 12, 8, 4, 2, 1):
        if num_channels % group_count == 0:
            return group_count
    return 1


class LocalConvAdapterBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.GroupNorm(_resolve_group_count(channels), channels)
        self.depthwise = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels)
        self.pointwise = nn.Conv2d(channels, channels, kernel_size=1)
        self.act = nn.GELU()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.norm(hidden_states)
        hidden_states = self.depthwise(hidden_states)
        hidden_states = self.pointwise(hidden_states)
        hidden_states = self.act(hidden_states)
        return residual + hidden_states


class NeighborhoodPatchMerger(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        merge_size: int = 2,
        init_std: float = 0.02,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.merge_size = merge_size
        merged_channels = in_channels * merge_size * merge_size
        self.norm = nn.LayerNorm(merged_channels)
        self.proj = nn.Linear(merged_channels, out_channels)
        nn.init.normal_(self.proj.weight, mean=0.0, std=init_std)
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, channels, height, width = hidden_states.shape
        if channels != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} channels, got {channels}.")
        if height % self.merge_size != 0 or width % self.merge_size != 0:
            raise ValueError(
                f"Spatial size {(height, width)} must be divisible by merge_size {self.merge_size}."
            )

        hidden_states = hidden_states.view(
            batch_size,
            channels,
            height // self.merge_size,
            self.merge_size,
            width // self.merge_size,
            self.merge_size,
        )
        hidden_states = hidden_states.permute(0, 2, 4, 3, 5, 1).contiguous()
        hidden_states = hidden_states.view(
            batch_size,
            height // self.merge_size,
            width // self.merge_size,
            channels * self.merge_size * self.merge_size,
        )
        hidden_states = self.norm(hidden_states)
        hidden_states = self.proj(hidden_states)
        return hidden_states.permute(0, 3, 1, 2).contiguous()


class MonkeyAdapterResamplerBridge(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        merge_size: int,
        target_grid: Tuple[int, int],
        num_local_blocks: int = 1,
        init_std: float = 0.02,
    ):
        super().__init__()
        self.merge_size = merge_size
        self.target_grid = target_grid
        self.local_blocks = nn.Sequential(
            *[LocalConvAdapterBlock(input_dim) for _ in range(max(0, num_local_blocks))]
        )
        self.merger = NeighborhoodPatchMerger(
            in_channels=input_dim,
            out_channels=output_dim,
            merge_size=merge_size,
            init_std=init_std,
        )
        self.pool = nn.AdaptiveAvgPool2d(target_grid)
        self.out_norm = nn.LayerNorm(output_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        grid_size: Tuple[int, int],
        token_order: str,
    ) -> torch.Tensor:
        hidden_states = token_sequence_to_grid(
            hidden_states,
            grid_size=grid_size,
            merge_size=self.merge_size,
            token_order=token_order,
        )
        hidden_states = self.local_blocks(hidden_states)
        hidden_states = self.merger(hidden_states)
        hidden_states = self.pool(hidden_states)
        hidden_states = hidden_states.flatten(2).transpose(1, 2).contiguous()
        return self.out_norm(hidden_states)
