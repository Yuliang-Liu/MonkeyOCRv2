#!/usr/bin/env python3
"""Build CDM evaluator input from MathWriting prediction JSONL."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


FUNC_NAMES = (
    "arccos",
    "arcsin",
    "arctan",
    "arg",
    "cosh",
    "coth",
    "csc",
    "deg",
    "det",
    "dim",
    "exp",
    "gcd",
    "hom",
    "inf",
    "ker",
    "liminf",
    "limsup",
    "lim",
    "ln",
    "log",
    "max",
    "min",
    "sec",
    "sinh",
    "sin",
    "sup",
    "tanh",
    "tan",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True, help="MathWriting prediction JSONL.")
    parser.add_argument("--output", required=True, help="Output CDM JSON list.")
    parser.add_argument("--limit", type=int, default=0, help="Optional first-N sample limit.")
    parser.add_argument(
        "--normalization",
        choices=("cdm", "none"),
        default="cdm",
        help="Use lightweight LaTeX cleanup for CDM rendering, or keep raw strings.",
    )
    parser.add_argument("--model-name", default="", help="Optional metadata only.")
    return parser.parse_args()


def read_jsonl(path: Path, limit: int = 0) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON") from exc
            if limit and len(rows) >= limit:
                break
    return rows


def strip_math_delimiters(text: str) -> str:
    text = text.strip()
    wrappers = (
        (r"\[", r"\]"),
        (r"\(", r"\)"),
        ("$$", "$$"),
        ("$", "$"),
    )
    changed = True
    while changed:
        changed = False
        for left, right in wrappers:
            if text.startswith(left) and text.endswith(right) and len(text) >= len(left) + len(right):
                text = text[len(left) : len(text) - len(right)].strip()
                changed = True
    return text


def find_matching(text: str, start: int, left: str, right: str) -> int:
    depth = 0
    i = start
    while i < len(text):
        char = text[i]
        if char == "\\":
            i += 2
            continue
        if char == left:
            depth += 1
        elif char == right:
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return -1


def convert_plain_sqrt(text: str) -> str:
    out: list[str] = []
    i = 0
    while i < len(text):
        if text.startswith("sqrt", i):
            prev = text[i - 1] if i > 0 else ""
            next_i = i + 4
            next_char = text[next_i] if next_i < len(text) else ""
            if prev != "\\" and not prev.isalpha() and not next_char.isalpha():
                j = next_i
                while j < len(text) and text[j].isspace():
                    j += 1
                if j < len(text) and text[j] in "({":
                    left = text[j]
                    right = ")" if left == "(" else "}"
                    end = find_matching(text, j, left, right)
                    if end != -1:
                        inner = convert_plain_sqrt(text[j + 1 : end])
                        out.append(r"\sqrt{" + inner.strip() + "}")
                        i = end + 1
                        continue
        out.append(text[i])
        i += 1
    return "".join(out)


def convert_plain_functions(text: str) -> str:
    names = "|".join(re.escape(name) for name in sorted(FUNC_NAMES, key=len, reverse=True))
    pattern = re.compile(rf"(?<![\\A-Za-z])({names})(?![A-Za-z])")
    return pattern.sub(lambda match: "\\" + match.group(1), text)


def normalize_for_cdm(text: Any) -> str:
    if text is None:
        return ""
    text = str(text)
    text = text.replace("\r", " ").replace("\n", " ").strip()
    text = strip_math_delimiters(text)
    text = re.sub(r"\\displaystyle\b", "", text)
    text = convert_plain_sqrt(text)
    text = convert_plain_functions(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def build_item(row: dict[str, Any], normalization: str, index: int) -> dict[str, Any]:
    sample_id = row.get("sample_id") or row.get("img_id") or f"sample_{index}"
    reference = row.get("reference", row.get("gt", ""))
    prediction = row.get("prediction", row.get("pred", ""))
    if normalization == "cdm":
        gt = normalize_for_cdm(reference)
        pred = normalize_for_cdm(prediction)
    else:
        gt = "" if reference is None else str(reference)
        pred = "" if prediction is None else str(prediction)
    item = {
        "img_id": str(sample_id),
        "gt": gt,
        "pred": pred,
    }
    image_path = row.get("crop_image") or row.get("image_path")
    if image_path:
        item["image_path"] = str(image_path)
    return item


def main() -> None:
    args = parse_args()
    rows = read_jsonl(Path(args.predictions), args.limit)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    items = [build_item(row, args.normalization, idx) for idx, row in enumerate(rows)]
    with output.open("w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)

    empty_gt = sum(1 for item in items if not item["gt"])
    empty_pred = sum(1 for item in items if not item["pred"])
    meta = f" for {args.model_name}" if args.model_name else ""
    print(f"Wrote {len(items)} CDM items{meta}: {output}")
    print(f"normalization={args.normalization}, empty_gt={empty_gt}, empty_pred={empty_pred}")


if __name__ == "__main__":
    main()
