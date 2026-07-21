#!/usr/bin/env python3
"""Static checks for a local MonkeyOCRv2 DFlash vLLM environment."""

from __future__ import annotations

import argparse
import importlib
import inspect
import os
import sys
from pathlib import Path


BACKENDS = ("FLASH_ATTN", "FLASHINFER")


def check(condition: bool, label: str, detail: str = "") -> bool:
    print(f"{'PASS' if condition else 'FAIL'} {label}{(': ' + detail) if detail else ''}")
    return condition


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backend", choices=BACKENDS, default="FLASHINFER")
    ap.add_argument("--model-path")
    ap.add_argument("--draft-model")
    args = ap.parse_args()

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

    try:
        import vllm
        print("vLLM version:", getattr(vllm, "__version__", "unknown"))
        print("vLLM path:", vllm.__file__)
        ok &= check(True, "vLLM import")
    except Exception as exc:
        ok &= check(False, "vLLM import", str(exc))

    try:
        from vllm.config import SpeculativeConfig
        signature = inspect.signature(SpeculativeConfig)
        has_method = "method" in signature.parameters and "dflash" in str(signature)
        ok &= check(has_method, "vLLM DFlash method", str(signature))
    except Exception as exc:
        ok &= check(False, "vLLM DFlash method", str(exc))

    try:
        from vllm.v1.attention.backends.registry import AttentionBackendEnum
        backend = getattr(AttentionBackendEnum, args.backend)
        ok &= check(True, f"Backend enum {args.backend}", str(backend))
    except Exception as exc:
        ok &= check(False, f"Backend enum {args.backend}", str(exc))

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
        ok &= check(present and supports_eagle3, "MonkeyOCRv2 vLLM plugin and SupportsEagle3")
    except Exception as exc:
        ok &= check(False, "MonkeyOCRv2 vLLM plugin import", str(exc))

    try:
        adapter = importlib.import_module("vllm.model_executor.models.qwen2_5_vl_dflash")
        ok &= check(
            hasattr(adapter, "Qwen25VLDFlashForConditionalGeneration"),
            "DFlash draft model adapter registration",
        )
    except Exception as exc:
        ok &= check(False, "DFlash draft model adapter import", str(exc))

    if args.backend == "FLASHINFER":
        try:
            flashinfer = importlib.import_module("flashinfer")
            ok &= check(True, "FlashInfer import", getattr(flashinfer, "__version__", "unknown"))
        except Exception as exc:
            ok &= check(False, "FlashInfer import", str(exc))
    else:
        try:
            interface = importlib.import_module("vllm.vllm_flash_attn.flash_attn_interface")
            fa2_available = bool(getattr(interface, "FA2_AVAILABLE", False))
            reason = getattr(interface, "FA2_UNAVAILABLE_REASON", "")
            ok &= check(fa2_available, "FlashAttention FA2 extension import", reason)
        except Exception as exc:
            ok &= check(False, "FlashAttention FA2 extension import", str(exc))
        print("INFO A successful import is not a runtime kernel validation; use validate_dflash_backend.py with a real image.")

    for label, value in (("target model", args.model_path), ("draft model", args.draft_model)):
        if value:
            ok &= check(Path(value).expanduser().is_dir(), label, str(Path(value).expanduser()))

    print("RUNTIME CHECK NOT PERFORMED: use validate_dflash_backend.py with a real image.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
