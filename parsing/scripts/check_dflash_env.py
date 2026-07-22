#!/usr/bin/env python3
"""Check a local vLLM, FlashAttention, and MonkeyOCRv2 DFlash environment."""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import sys
from pathlib import Path


def check(condition: bool, label: str, detail: str = "") -> bool:
    suffix = f": {detail}" if detail else ""
    print(f"{'PASS' if condition else 'FAIL'} {label}{suffix}")
    return condition


def load_config(model_dir: Path) -> dict:
    config_path = model_dir / "config.json"
    if not config_path.is_file():
        raise ValueError(f"missing config.json: {model_dir}")
    return json.loads(config_path.read_text(encoding="utf-8"))


def resolve_text_config(config: dict) -> dict:
    for key in ("text_config", "language_config", "llm_config"):
        nested = config.get(key)
        if isinstance(nested, dict):
            return nested
    return config


def check_model_compatibility(target_dir: Path, draft_dir: Path) -> list[str]:
    errors: list[str] = []
    if not target_dir.is_dir() or not draft_dir.is_dir():
        return ["target and draft must both be model directories"]
    if target_dir.resolve() == draft_dir.resolve():
        return ["target and draft must be different directories"]
    try:
        target, draft = load_config(target_dir), load_config(draft_dir)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return [str(exc)]

    target_text = resolve_text_config(target)
    draft_text = resolve_text_config(draft)
    target_arch = target.get("architectures", [])
    if not any("MonkeyOCRv2" in str(name) for name in target_arch):
        errors.append(f"target is not a MonkeyOCRv2 parsing model: {target_arch}")

    draft_arch = draft.get("architectures", [])
    resolved_arch = [
        arch if str(arch).startswith("DFlash") else f"DFlash{arch}"
        for arch in draft_arch
    ]
    if "DFlashMonkeyOCRv2ForCausalLM" not in resolved_arch:
        errors.append(
            "draft architecture must resolve to "
            "DFlashMonkeyOCRv2ForCausalLM: "
            f"architectures={draft_arch}"
        )

    dflash_config = draft.get("dflash_config")
    if not isinstance(dflash_config, dict):
        errors.append("draft config has no dflash_config")
    else:
        training_config = dflash_config.get("training_config")
        if not isinstance(training_config, dict):
            training_config = {}
        block_size = dflash_config.get(
            "block_size",
            draft.get("block_size", training_config.get("block_size")),
        )
        if block_size != 16:
            errors.append(f"draft DFlash block_size must be 16, got {block_size}")

    target_vocab = target_text.get("vocab_size")
    draft_vocab = draft.get("draft_vocab_size")
    if draft_vocab is None:
        draft_vocab = draft_text.get("draft_vocab_size", draft_text.get("vocab_size"))
    if target_vocab is None or target_vocab != draft_vocab:
        errors.append(f"vocab_size mismatch: target={target_vocab}, draft={draft_vocab}")
    for key in ("eos_token_id", "pad_token_id", "bos_token_id", "image_token_id", "video_token_id"):
        target_value = target.get(key, target_text.get(key))
        draft_value = draft.get(key, draft_text.get(key))
        if target_value is not None and draft_value is not None and target_value != draft_value:
            errors.append(f"{key} mismatch: target={target_value}, draft={draft_value}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-model")
    parser.add_argument("--draft-model")
    parser.add_argument("--vllm-source")
    parser.add_argument(
        "--require-native-dflash",
        action="store_true",
        help="Require DFlash to come from the installed vLLM package.",
    )
    args = parser.parse_args()

    if args.vllm_source:
        sys.path.insert(0, str(Path(args.vllm_source).expanduser().resolve()))

    print("STATIC CHECK")
    print("Python:", sys.version.split()[0])
    ok = True
    try:
        import torch

        ok &= check(True, "PyTorch import", torch.__version__)
        ok &= check(torch.cuda.is_available(), "CUDA available")
        if torch.cuda.is_available():
            print("GPU count:", torch.cuda.device_count())
            print("GPU 0:", torch.cuda.get_device_name(0))
    except Exception as exc:
        ok &= check(False, "PyTorch/CUDA import", str(exc))

    vllm_path = None
    try:
        import vllm

        version = getattr(vllm, "__version__", "unknown")
        print("vLLM version:", version)
        print("vLLM path:", vllm.__file__)
        vllm_path = Path(vllm.__file__).resolve()
        if args.require_native_dflash and args.vllm_source:
            ok &= check(False, "Native vLLM package", "--vllm-source cannot be used")
        elif args.require_native_dflash:
            ok &= check(True, "Native vLLM package", str(vllm_path))
            ok &= check(version == "0.25.1", "Native vLLM version", version)
        ok &= check(True, "vLLM import")
    except Exception as exc:
        ok &= check(False, "vLLM import", str(exc))

    try:
        from vllm.config import SpeculativeConfig

        supports_dflash = "dflash" in inspect.getsource(SpeculativeConfig).lower()
        ok &= check(supports_dflash, "vLLM method=dflash support")
    except Exception as exc:
        ok &= check(False, "vLLM method=dflash support", str(exc))

    try:
        from vllm.v1.attention.backends.registry import AttentionBackendEnum

        getattr(AttentionBackendEnum, "FLASH_ATTN")
        ok &= check(True, "Backend enum FLASH_ATTN")
    except Exception as exc:
        ok &= check(False, "Backend enum FLASH_ATTN", str(exc))

    try:
        dflash = importlib.import_module("vllm.v1.spec_decode.dflash")
        ok &= check(hasattr(dflash, "DFlashProposer"), "DFlash proposer import")
    except Exception as exc:
        ok &= check(False, "DFlash proposer import", str(exc))

    parsing_dir = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(parsing_dir))
    try:
        plugin = importlib.import_module("modeling.modeling_monkeyocrv2_vllm")
        present = hasattr(plugin, "MonkeyOCRv2ForCausalLM")
        supports_eagle3 = "SupportsEagle3" in inspect.getsource(plugin)
        ok &= check(present and supports_eagle3, "MonkeyOCRv2 plugin and SupportsEagle3")
    except Exception as exc:
        ok &= check(False, "MonkeyOCRv2 plugin import", str(exc))

    try:
        adapter = importlib.import_module(
            "modeling.modeling_monkeyocrv2_dflash_vllm"
        )
        ok &= check(
            hasattr(adapter, "DFlashMonkeyOCRv2ForCausalLM"),
            "MonkeyOCRv2 DFlash draft adapter import",
        )
    except Exception as exc:
        ok &= check(False, "MonkeyOCRv2 DFlash draft adapter import", str(exc))

    try:
        model_cls = plugin.MonkeyOCRv2ForCausalLM
        ok &= check(
            isinstance(getattr(model_cls, "lm_head", None), property),
            "Target lm_head sharing property",
        )
    except Exception as exc:
        ok &= check(False, "Target lm_head sharing property", str(exc))

    try:
        interface = importlib.import_module("vllm.vllm_flash_attn.flash_attn_interface")
        available = bool(getattr(interface, "FA2_AVAILABLE", False))
        reason = getattr(interface, "FA2_UNAVAILABLE_REASON", "")
        ok &= check(available, "FlashAttention FA2 extension import", reason)
    except Exception as exc:
        ok &= check(False, "FlashAttention FA2 extension import", str(exc))

    for label, value in (("target model", args.target_model), ("draft model", args.draft_model)):
        if value:
            ok &= check(Path(value).expanduser().is_dir(), label)
    if args.target_model and args.draft_model:
        errors = check_model_compatibility(
            Path(args.target_model).expanduser(), Path(args.draft_model).expanduser()
        )
        ok &= check(not errors, "Target/DFlash model compatibility", "; ".join(errors))

    print("STATIC CHECK PASSED" if ok else "STATIC CHECK FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
