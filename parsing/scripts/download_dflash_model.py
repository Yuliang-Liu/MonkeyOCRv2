#!/usr/bin/env python3
"""Download or copy a MonkeyOCRv2 DFlash draft model."""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path


def verify_model_dir(model_dir: Path) -> list[str]:
    missing: list[str] = []
    if not (model_dir / "config.json").is_file():
        missing.append("config.json")
    if not (model_dir / "tokenizer_config.json").is_file() and not (
        model_dir / "tokenizer.json"
    ).is_file():
        missing.append("tokenizer_config.json or tokenizer.json")
    has_weights = any(
        p.is_file()
        and (p.name.endswith(".safetensors") or p.name.endswith(".bin"))
        for p in model_dir.rglob("*")
    )
    if not has_weights and not (model_dir / "model.safetensors.index.json").is_file():
        missing.append("safetensors or bin weights")
    return missing


def download_hf(args, output_dir: Path) -> None:
    from huggingface_hub import snapshot_download

    token = os.getenv(args.token_env) if args.token_env else None
    snapshot_download(
        repo_id=args.repo_id,
        local_dir=str(output_dir),
        revision=args.revision,
        token=token,
    )


def download_modelscope(args, output_dir: Path) -> None:
    try:
        from modelscope.hub.snapshot_download import snapshot_download
    except ImportError as exc:
        raise SystemExit(
            "ModelScope download requires an already installed modelscope client."
        ) from exc

    snapshot_download(
        model_id=args.repo_id,
        local_dir=str(output_dir),
        revision=args.revision,
    )


def copy_local(args, output_dir: Path) -> None:
    source_dir = Path(args.local_source).expanduser().resolve()
    if not source_dir.is_dir():
        raise SystemExit(f"Local source does not exist: {source_dir}")
    shutil.copytree(source_dir, output_dir, dirs_exist_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=("hf", "modelscope", "local"), required=True)
    parser.add_argument("--repo-id")
    parser.add_argument("--local-source")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--token-env", default="HF_TOKEN")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()

    if args.source in {"hf", "modelscope"} and not args.repo_id:
        parser.error("--repo-id is required for hf and modelscope")
    if args.source == "local" and not args.local_source:
        parser.error("--local-source is required for local mode")

    output_dir = Path(args.output_dir).expanduser().resolve()
    if args.verify_only:
        missing = verify_model_dir(output_dir)
        if missing:
            print("MODEL CHECK FAILED:")
            for item in missing:
                print(f"- {item}")
            return 1
        print(f"MODEL CHECK PASSED: {output_dir}")
        return 0

    if output_dir.exists() and any(output_dir.iterdir()):
        if not args.force:
            raise SystemExit(
                f"Refusing to overwrite non-empty directory: {output_dir}; use --force."
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.source == "hf":
        download_hf(args, output_dir)
    elif args.source == "modelscope":
        download_modelscope(args, output_dir)
    else:
        copy_local(args, output_dir)

    missing = verify_model_dir(output_dir)
    if missing:
        print("MODEL CHECK FAILED:")
        for item in missing:
            print(f"- {item}")
        return 1
    print(f"MODEL DOWNLOAD/COPY PASSED: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
