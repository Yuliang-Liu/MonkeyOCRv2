#!/usr/bin/env python3
"""Validate target and DFlash draft model configuration compatibility."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_config(model_dir: Path) -> dict:
    path = model_dir / "config.json"
    if not path.is_file():
        raise ValueError(f"missing config.json: {model_dir}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid config.json: {model_dir}: {exc}") from exc


def architectures(config: dict) -> list[str]:
    value = config.get("architectures", [])
    return [str(item) for item in value] if isinstance(value, list) else []


def check_models(target_dir: Path, draft_dir: Path, expected_block_size: int) -> list[str]:
    errors: list[str] = []
    if not target_dir.is_dir():
        errors.append(f"target directory does not exist: {target_dir}")
        return errors
    if not draft_dir.is_dir():
        errors.append(f"draft directory does not exist: {draft_dir}")
        return errors
    if target_dir.resolve() == draft_dir.resolve():
        errors.append("target and draft must be different directories")
        return errors

    try:
        target = load_config(target_dir)
        draft = load_config(draft_dir)
    except ValueError as exc:
        return [str(exc)]

    target_arch = architectures(target)
    draft_arch = architectures(draft)
    if not any("MonkeyOCRv2" in name for name in target_arch):
        errors.append(f"target is not a MonkeyOCRv2 parsing model: {target_arch}")

    dflash_config = draft.get("dflash_config")
    if not isinstance(dflash_config, dict):
        errors.append("draft config has no dflash_config")
    elif dflash_config.get("block_size") != expected_block_size:
        errors.append(
            "draft dflash_config.block_size must be "
            f"{expected_block_size}, got {dflash_config.get('block_size')!r}"
        )

    if not any("DFlash" in name for name in draft_arch) and not dflash_config:
        errors.append(f"draft has no DFlash architecture marker: {draft_arch}")

    target_vocab = target.get("vocab_size")
    draft_vocab = draft.get("draft_vocab_size", draft.get("vocab_size"))
    if target_vocab is None or draft_vocab is None:
        errors.append("target/draft vocab_size is missing")
    elif target_vocab != draft_vocab:
        errors.append(f"vocab_size mismatch: target={target_vocab}, draft={draft_vocab}")

    for key in ("eos_token_id", "pad_token_id", "bos_token_id", "image_token_id", "video_token_id"):
        target_value = target.get(key)
        draft_value = draft.get(key)
        if target_value is not None and draft_value is not None and target_value != draft_value:
            errors.append(f"{key} mismatch: target={target_value}, draft={draft_value}")

    target_model_type = str(target.get("model_type", ""))
    draft_model_type = str(draft.get("model_type", ""))
    if target_model_type == draft_model_type and not dflash_config:
        errors.append("draft model_type is indistinguishable from target without DFlash config")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-model", required=True)
    parser.add_argument("--draft-model", required=True)
    parser.add_argument("--expected-block-size", type=int, default=16)
    args = parser.parse_args()

    errors = check_models(
        Path(args.target_model).expanduser(),
        Path(args.draft_model).expanduser(),
        args.expected_block_size,
    )
    if errors:
        print("MODEL COMPATIBILITY CHECK FAILED")
        for error in errors:
            print(f"- {error}")
        return 1
    print("MODEL COMPATIBILITY CHECK PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
