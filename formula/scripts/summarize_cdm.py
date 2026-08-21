#!/usr/bin/env python3
"""Summarize CDM metric outputs."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="cdm_outputs", help="Directory with */metrics_res.json.")
    parser.add_argument("--input-root", default="cdm_inputs", help="Directory with CDM input JSON files.")
    parser.add_argument("--output", default="reports/cdm_summary.json", help="Output JSON summary.")
    parser.add_argument("--markdown-output", default="reports/cdm_summary.md", help="Output Markdown table.")
    parser.add_argument("--title", default="CDM Summary", help="Markdown title.")
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def count_input_rows(path: Path) -> int | None:
    if not path.exists():
        return None
    data = load_json(path)
    if isinstance(data, list):
        return len(data)
    return None


def split_name(name: str) -> tuple[str, str]:
    parts = name.split("__", 1)
    if len(parts) == 2:
        return parts[0], parts[1]
    return "unknown", name


def finite_or_none(value: Any) -> float | None:
    if value is None:
        return None
    value = float(value)
    if math.isfinite(value):
        return value
    return None


def collect_rows(root: Path, input_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for metrics_path in sorted(root.glob("*/metrics_res.json")):
        name = metrics_path.parent.name
        model, subset = split_name(name)
        metrics = load_json(metrics_path)
        input_count = count_input_rows(input_root / f"{name}.json")
        num_details = len(metrics.get("details", {}))
        render_coverage = None
        if input_count:
            render_coverage = num_details / input_count
        rows.append(
            {
                "name": name,
                "model": model,
                "subset": subset,
                "input_count": input_count,
                "num_details": num_details,
                "render_coverage": render_coverage,
                "mean_score": finite_or_none(metrics.get("mean_score")),
                "exp_rate": finite_or_none(metrics.get("exp_rate")),
                "metrics_path": str(metrics_path),
            }
        )
    return rows


def write_markdown(rows: list[dict[str, Any]], path: Path, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# {title}",
        "",
        "CDM is rendered-LaTeX visual matching. Exact rate is the expression-level exact rendering rate.",
        "",
        "| model | subset | input | scored | render coverage | mean F1 | exact rate |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        input_count = row["input_count"]
        input_text = "" if input_count is None else str(input_count)
        coverage = row["render_coverage"]
        coverage_text = "" if coverage is None else f"{coverage:.4f}"
        mean_score = finite_or_none(row["mean_score"])
        mean_text = "" if mean_score is None else f"{float(mean_score):.3f}"
        exp_rate = finite_or_none(row["exp_rate"])
        exp_text = "" if exp_rate is None else f"{float(exp_rate):.3f}"
        lines.append(
            f"| {row['model']} | {row['subset']} | {input_text} | {row['num_details']} | "
            f"{coverage_text} | {mean_text} | {exp_text} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    rows = collect_rows(Path(args.root), Path(args.input_root))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2, allow_nan=False)
    write_markdown(rows, Path(args.markdown_output), args.title)
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
