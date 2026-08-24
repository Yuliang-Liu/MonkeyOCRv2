#!/usr/bin/env python3
"""Score MathWriting predictions with the paper's LaTeX-token CER."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Sequence


TOKEN_RE = re.compile(r"(\\[a-zA-Z]+)|([a-zA-Z0-9])|(\S)")


def tokenize_latex(text: str) -> list[str]:
    tokens: list[str] = []
    for command, alnum, other in TOKEN_RE.findall(text):
        tokens.append(command or alnum or other)
    return tokens


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
        for left, right in pairs:
            if text.startswith(left) and text.endswith(right) and len(text) >= len(left) + len(right):
                text = text[len(left) : len(text) - len(right)].strip()
                changed = True
                break
    return text


def normalize_for_scoring(text: str, strip_delimiters: bool) -> str:
    text = text.strip()
    if strip_delimiters:
        text = strip_latex_delimiters(text)
    return text


def edit_distance(left: Sequence[Any], right: Sequence[Any]) -> int:
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for i, left_item in enumerate(left, start=1):
        current = [i]
        for j, right_item in enumerate(right, start=1):
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + (left_item != right_item),
                )
            )
        previous = current
    return previous[-1]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_manifest_references(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    refs = {}
    for row in read_jsonl(path):
        refs[row["sample_id"]] = row.get("reference", "")
    return refs


def compute_bleu(preds: list[str], refs: list[str], repo_root: Path | None) -> float | None:
    if repo_root is not None:
        sys.path.insert(0, str(repo_root.resolve()))
        try:
            from unimernet.common.metrics import compute_corpus_bleu

            return float(compute_corpus_bleu(preds, refs))
        except Exception:
            pass
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--repo-root", type=Path, default=None)
    parser.add_argument("--per-sample-output", type=Path, default=None)
    parser.add_argument("--keep-delimiters", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pred_rows = read_jsonl(args.predictions)
    manifest_refs = load_manifest_references(args.manifest)
    strip_delimiters = not args.keep_delimiters

    per_sample = []
    total_token_distance = 0
    total_reference_tokens = 0
    total_char_distance = 0
    total_reference_chars = 0
    exact_string = 0
    exact_tokens = 0
    preds_for_bleu = []
    refs_for_bleu = []

    for row in pred_rows:
        sample_id = row["sample_id"]
        ref_raw = row.get("reference") or manifest_refs.get(sample_id, "")
        pred_raw = row.get("prediction", "")
        ref = normalize_for_scoring(ref_raw, strip_delimiters)
        pred = normalize_for_scoring(pred_raw, strip_delimiters)
        ref_tokens = tokenize_latex(ref)
        pred_tokens = tokenize_latex(pred)
        token_distance = edit_distance(pred_tokens, ref_tokens)
        char_distance = edit_distance(pred, ref)

        total_token_distance += token_distance
        total_reference_tokens += len(ref_tokens)
        total_char_distance += char_distance
        total_reference_chars += len(ref)
        exact_string += int(pred == ref)
        exact_tokens += int(pred_tokens == ref_tokens)
        preds_for_bleu.append(pred)
        refs_for_bleu.append(ref)
        per_sample.append(
            {
                "sample_id": sample_id,
                "split": row.get("split", ""),
                "reference": ref,
                "prediction": pred,
                "token_distance": token_distance,
                "reference_tokens": len(ref_tokens),
                "token_cer": token_distance / len(ref_tokens) if ref_tokens else None,
                "char_distance": char_distance,
                "reference_chars": len(ref),
                "char_cer": char_distance / len(ref) if ref else None,
                "exact_string": pred == ref,
                "exact_tokens": pred_tokens == ref_tokens,
            }
        )

    num_samples = len(per_sample)
    report = {
        "num_samples": num_samples,
        "mathwriting_token_cer": (
            total_token_distance / total_reference_tokens if total_reference_tokens else None
        ),
        "total_token_distance": total_token_distance,
        "total_reference_tokens": total_reference_tokens,
        "char_cer": total_char_distance / total_reference_chars if total_reference_chars else None,
        "total_char_distance": total_char_distance,
        "total_reference_chars": total_reference_chars,
        "exact_string_accuracy": exact_string / num_samples if num_samples else None,
        "exact_token_accuracy": exact_tokens / num_samples if num_samples else None,
        "bleu": compute_bleu(preds_for_bleu, refs_for_bleu, args.repo_root),
        "strip_latex_delimiters": strip_delimiters,
        "metric_note": "Primary MathWriting metric is LaTeX-token CER: total edit distance over tokenized references.",
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    if args.per_sample_output is not None:
        args.per_sample_output.parent.mkdir(parents=True, exist_ok=True)
        with args.per_sample_output.open("w", encoding="utf-8") as f:
            for row in per_sample:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
