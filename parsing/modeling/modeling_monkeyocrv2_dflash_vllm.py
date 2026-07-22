# SPDX-License-Identifier: Apache-2.0
"""MonkeyOCRv2 DFlash draft adapter for vLLM.

MonkeyOCRv2 parsing uses a multimodal target model. A DFlash draft does not
need to execute the visual tower: the
target model owns image encoding and provides hidden states to the draft through
vLLM's speculative-decoding interface. This adapter therefore reuses the
existing text DFlash core and only handles the VL-specific config, MRoPE
positions, and weight-name prefixes.

This file is an adapter layer, not a trained checkpoint. End-to-end DFlash
serving still requires a Qwen2.5-VL/MonkeyOCR-compatible DFlash draft checkpoint
whose weights match the DFlash core.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch
from transformers import PretrainedConfig

from vllm.config import VllmConfig
from vllm.model_executor.models.qwen3_dflash import DFlashQwen3ForCausalLM
from vllm.multimodal.inputs import NestedTensors


def _select_text_positions(positions: torch.Tensor) -> torch.Tensor:
    """Convert Qwen2.5-VL MRoPE positions to text positions for DFlash."""

    if isinstance(positions, torch.Tensor) and positions.dim() == 2:
        if positions.shape[0] == 3:
            return positions[0]
    return positions


def _as_text_dflash_config(vllm_config: VllmConfig):
    """Expose a Qwen2.5-VL language config to the text DFlash core."""

    draft_model_config = vllm_config.speculative_config.draft_model_config
    original_hf_config = draft_model_config.hf_config
    text_config = (
        getattr(original_hf_config, "text_config", None)
        or getattr(original_hf_config, "language_config", None)
        or getattr(original_hf_config, "llm_config", None)
    )
    if isinstance(text_config, dict):
        text_config = PretrainedConfig(**text_config)
    if text_config is None:
        text_config = original_hf_config

    # Older training exports store the DFlash mask token in the nested training
    # section. vLLM's runtime and the DFlash core both consume the normalized
    # ``dflash_config.mask_token_id`` field.
    dflash_config = getattr(original_hf_config, "dflash_config", None) or {}
    training_config = dflash_config.get("training_config", {})
    mask_token_id = dflash_config.get("mask_token_id")
    if mask_token_id is None:
        mask_token_id = training_config.get("mask_token_id")
        if mask_token_id is not None:
            dflash_config = {**dflash_config, "mask_token_id": mask_token_id}
            original_hf_config.dflash_config = dflash_config
    if mask_token_id is not None and getattr(original_hf_config, "mask_token_id", None) is None:
        original_hf_config.mask_token_id = mask_token_id

    for name in (
        "vocab_size", "hidden_size", "intermediate_size", "hidden_act",
        "num_hidden_layers", "num_attention_heads", "num_key_value_heads",
        "head_dim", "rms_norm_eps", "max_position_embeddings",
        "rope_theta", "rope_scaling", "rope_parameters",
        "tie_word_embeddings", "bos_token_id", "eos_token_id",
        "pad_token_id", "sliding_window", "use_sliding_window",
        "layer_types", "qk_norm",
    ):
        if hasattr(original_hf_config, name) and not hasattr(text_config, name):
            setattr(text_config, name, getattr(original_hf_config, name))
        elif hasattr(text_config, name) and not hasattr(original_hf_config, name):
            setattr(original_hf_config, name, getattr(text_config, name))

    for name in (
        "dflash_config", "eagle_config", "draft_vocab_size",
        "logit_scale", "target_hidden_size",
    ):
        if hasattr(original_hf_config, name) and not hasattr(text_config, name):
            setattr(text_config, name, getattr(original_hf_config, name))

    num_draft_layers = dflash_config.get("num_draft_layers")
    if num_draft_layers is not None:
        text_config.num_hidden_layers = int(num_draft_layers)

    draft_model_config.hf_config = text_config
    return original_hf_config, text_config


def _restore_hf_config(vllm_config: VllmConfig, original_hf_config) -> None:
    vllm_config.speculative_config.draft_model_config.hf_config = original_hf_config


class Qwen25VLDFlashForConditionalGeneration(DFlashQwen3ForCausalLM):
    """DFlash draft adapter for Qwen2.5-VL / MonkeyOCR-Pro Recognition."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        original_hf_config, text_config = _as_text_dflash_config(vllm_config)
        try:
            super().__init__(vllm_config=vllm_config, prefix=prefix)
        finally:
            _restore_hf_config(vllm_config, original_hf_config)
        self.vl_config = original_hf_config
        self.text_config = text_config

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: NestedTensors | None = None,
        is_multimodal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.model(input_ids, _select_text_positions(positions), inputs_embeds)

    def precompute_and_store_context_kv(
        self,
        context_states: torch.Tensor,
        context_positions: torch.Tensor,
        context_slot_mapping: torch.Tensor | list[torch.Tensor | None] | None = None,
    ) -> None:
        self.model.precompute_and_store_context_kv(
            context_states,
            _select_text_positions(context_positions),
            context_slot_mapping,
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        """Load draft weights while skipping target-only visual tower weights."""

        weights = list(weights)
        has_explicit_draft = any(
            name.startswith(("draft_model.", "model.draft_model."))
            for name, _ in weights
        )
        remapped: list[tuple[str, torch.Tensor]] = []
        skip_prefixes = (
            "visual.", "model.visual.", "vision_tower.", "model.vision_tower.",
        )
        target_prefixes = (
            "model.language_model.", "language_model.",
            "model.model.language_model.",
        )
        strip_prefixes = (
            "draft_model.", "model.draft_model.",
            "model.language_model.", "language_model.",
            "model.model.language_model.",
        )

        for name, tensor in weights:
            if name.startswith(skip_prefixes):
                continue
            if has_explicit_draft and name.startswith(target_prefixes):
                continue
            for prefix in strip_prefixes:
                if name.startswith(prefix):
                    name = name[len(prefix):]
                    break
            remapped.append((name, tensor))

        return super().load_weights(remapped)


DFlashQwen25VLForConditionalGeneration = Qwen25VLDFlashForConditionalGeneration
MonkeyOCRv2DFlashForConditionalGeneration = Qwen25VLDFlashForConditionalGeneration
DFlashQwen25VLDFlashForConditionalGeneration = Qwen25VLDFlashForConditionalGeneration
DFlashMonkeyOCRv2ForCausalLM = Qwen25VLDFlashForConditionalGeneration
