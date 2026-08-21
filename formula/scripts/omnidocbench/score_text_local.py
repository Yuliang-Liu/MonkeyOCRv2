#!/usr/bin/env python3
"""Local Edit/BLEU sanity scoring for prediction JSONL.

This does not replace the official OmniDocBench evaluator, especially for CDM.
It exists to catch obvious prediction/config mistakes before running the slower
official pipeline.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

from rapidfuzz.distance import Levenshtein


def normalize_text(text: str) -> str:
    text = strip_latex_delimiters(text)
    text_reg = r"(\\(operatorname|mathrm|text|mathbf)\s?\*? {.*?})"
    letter = "[a-zA-Z]"
    noletter = r"[\W_^\d]"
    names = [x[0].replace(" ", "") for x in re.findall(text_reg, text)]
    text = re.sub(text_reg, lambda _match: str(names.pop(0)), text)
    news = text
    while True:
        text = news
        news = re.sub(r"(?!\\ )(%s)\s+?(%s)" % (noletter, noletter), r"\1\2", text)
        news = re.sub(r"(?!\\ )(%s)\s+?(%s)" % (noletter, letter), r"\1\2", news)
        news = re.sub(r"(%s)\s+?(%s)" % (letter, noletter), r"\1\2", news)
        if news == text:
            break
    return text


def strip_latex_delimiters(text: str) -> str:
    text = text.strip()
    pairs = (
        ("$$", "$$"),
        ("$", "$"),
        (r"\[", r"\]"),
        (r"\(", r"\)"),
    )
    changed = True
    while changed:
        changed = False
        text = text.strip()
        for left, right in pairs:
            if text.startswith(left) and text.endswith(right) and len(text) >= len(left) + len(right):
                text = text[len(left) : len(text) - len(right)]
                changed = True
                break
    return text.strip()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def compute_bleu(preds: list[str], refs: list[str], repo_root: Path | None) -> float | None:
    if repo_root is not None:
        sys.path.insert(0, str(repo_root))
        try:
            from unimernet.common.metrics import compute_corpus_bleu

            return float(compute_corpus_bleu(preds, refs))
        except Exception:
            pass
    try:
        import sacrebleu

        return float(sacrebleu.corpus_bleu(preds, [refs], tokenize="none").score / 100.0)
    except Exception:
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--repo-root", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.predictions)
    preds = [normalize_text(row.get("prediction", "")) for row in rows]
    refs = [normalize_text(row.get("reference", "")) for row in rows]
    edits = [Levenshtein.normalized_distance(pred, ref) for pred, ref in zip(preds, refs)]
    report = {
        "num_samples": len(rows),
        "edit": sum(edits) / len(edits) if edits else None,
        "bleu": compute_bleu(preds, refs, args.repo_root),
        "note": "Local sanity metrics only; use official OmniDocBench for final CDM/Edit/BLEU.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
