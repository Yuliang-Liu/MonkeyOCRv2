#!/usr/bin/env python3
"""Merge JSONL predictions into OmniDocBench JSON as a `pred` field."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def parse_pointer(pointer: str) -> list[str | int]:
    if not pointer.startswith("/"):
        raise ValueError(f"Invalid JSON pointer: {pointer}")
    parts: list[str | int] = []
    for part in pointer[1:].split("/"):
        part = part.replace("~1", "/").replace("~0", "~")
        if part.isdigit():
            parts.append(int(part))
        else:
            parts.append(part)
    return parts


def resolve_pointer(data: Any, pointer: str) -> Any:
    current = data
    for part in parse_pointer(pointer):
        current = current[part]
    return current


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--prediction-field", default="pred")
    parser.add_argument("--empty-missing", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = load_json(args.annotation)
    manifest_rows = {row["sample_id"]: row for row in read_jsonl(args.manifest)}
    prediction_rows = {row["sample_id"]: row for row in read_jsonl(args.predictions)}

    merged = 0
    missing = []
    for sample_id, manifest in manifest_rows.items():
        node = resolve_pointer(data, manifest["json_pointer"])
        pred_row = prediction_rows.get(sample_id)
        if pred_row is None:
            missing.append(sample_id)
            if args.empty_missing:
                node[args.prediction_field] = ""
            continue
        node[args.prediction_field] = pred_row.get("prediction", "")
        merged += 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)

    print(
        json.dumps(
            {
                "output": str(args.output),
                "merged": merged,
                "missing_predictions": len(missing),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

