#!/usr/bin/env python3
"""Download a DFlash draft from Hugging Face or ModelScope into a local directory."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=("hf", "modelscope"), required=True)
    parser.add_argument("--repo-id", required=True, help="Repository id, for example org/MonkeyOCRv2-B-Parsing-DFlash")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--revision", default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    if args.source == "hf":
        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id=args.repo_id,
            local_dir=str(output_dir),
            revision=args.revision,
        )
    else:
        try:
            from modelscope.hub.snapshot_download import snapshot_download
        except ImportError as exc:
            raise SystemExit("ModelScope download needs `pip install modelscope`.") from exc

        snapshot_download(
            model_id=args.repo_id,
            cache_dir=str(output_dir.parent),
            revision=args.revision,
            local_dir=str(output_dir),
        )

    print(output_dir)


if __name__ == "__main__":
    main()
