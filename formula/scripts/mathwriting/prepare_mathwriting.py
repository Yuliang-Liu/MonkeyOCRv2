#!/usr/bin/env python3
"""Render MathWriting InkML files to PNG and build UniMERNet manifests."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import xml.etree.ElementTree as ET
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw
from tqdm import tqdm


ANNOTATION_KEYS = (
    "normalizedLabel",
    "label",
    "truth",
    "groundTruth",
)


@dataclass
class InkSample:
    inkml_path: Path
    split: str
    sample_id: str
    label: str
    traces: list[list[tuple[float, float]]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--splits", nargs="+", default=["test"])
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--image-dir", required=True, type=Path)
    parser.add_argument("--stroke-width", type=float, default=1.5)
    parser.add_argument("--margin", type=float, default=10.0)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--antialias", type=int, default=4)
    parser.add_argument("--max-render-side", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def maybe_dataset_root(root: Path) -> Path:
    root = root.resolve()
    if any((root / split).is_dir() for split in ("test", "valid", "train", "synthetic")):
        return root
    children = [p for p in root.iterdir() if p.is_dir()] if root.exists() else []
    for child in children:
        if any((child / split).is_dir() for split in ("test", "valid", "train", "synthetic")):
            return child.resolve()
    return root


def split_for_path(path: Path, root: Path, requested_splits: set[str]) -> str | None:
    try:
        parts = path.relative_to(root).parts
    except ValueError:
        parts = path.parts
    for part in parts:
        if part in requested_splits:
            return part
    return None


def read_label(root: ET.Element) -> str:
    fallback = ""
    for node in root.iter():
        if local_name(node.tag) != "annotation":
            continue
        value = (node.text or "").strip()
        if not value:
            continue
        annotation_type = node.attrib.get("type") or node.attrib.get("encoding") or ""
        if annotation_type in ANNOTATION_KEYS:
            return value
        if not fallback:
            fallback = value
    return fallback


def parse_trace_text(text: str | None) -> list[tuple[float, float]]:
    if not text:
        return []
    points: list[tuple[float, float]] = []
    for chunk in text.replace("\n", " ").split(","):
        nums = re.findall(r"[-+]?(?:\d+\.\d+|\d+|\.\d+)(?:[eE][-+]?\d+)?", chunk)
        if len(nums) >= 2:
            points.append((float(nums[0]), float(nums[1])))
    return points


def parse_inkml(path: Path, root_dir: Path, requested_splits: set[str]) -> InkSample | None:
    split = split_for_path(path, root_dir, requested_splits)
    if split is None:
        return None
    tree = ET.parse(path)
    xml_root = tree.getroot()
    label = read_label(xml_root)
    traces = [
        trace
        for node in xml_root.iter()
        if local_name(node.tag) == "trace"
        for trace in [parse_trace_text(node.text)]
        if trace
    ]
    if not traces or not label:
        return None
    stem = path.stem
    sample_id = f"{split}_{stem}"
    return InkSample(path, split, sample_id, label, traces)


def iter_samples(root: Path, splits: list[str]) -> Iterable[InkSample]:
    requested_splits = set(splits)
    for split in splits:
        split_dir = root / split
        if split_dir.is_dir():
            paths = sorted(split_dir.glob("*.inkml"))
        else:
            paths = [p for p in sorted(root.rglob("*.inkml")) if split_for_path(p, root, requested_splits) == split]
        for path in paths:
            sample = parse_inkml(path, root, requested_splits)
            if sample is not None:
                yield sample


def trace_bbox(traces: list[list[tuple[float, float]]]) -> tuple[float, float, float, float]:
    xs = [x for trace in traces for x, _ in trace]
    ys = [y for trace in traces for _, y in trace]
    return min(xs), min(ys), max(xs), max(ys)


def render_sample(
    sample: InkSample,
    output_path: Path,
    stroke_width: float,
    margin: float,
    scale: float,
    antialias: int,
    max_render_side: int,
) -> None:
    min_x, min_y, max_x, max_y = trace_bbox(sample.traces)
    width_units = max(1.0, max_x - min_x + margin * 2)
    height_units = max(1.0, max_y - min_y + margin * 2)
    draw_scale = scale
    if max_render_side > 0:
        draw_scale = min(draw_scale, max_render_side / max(width_units, height_units))
    draw_scale = max(draw_scale, 1e-6)
    aa = max(1, antialias)

    full_scale = draw_scale * aa
    canvas_w = max(1, int(math.ceil(width_units * full_scale)))
    canvas_h = max(1, int(math.ceil(height_units * full_scale)))
    image = Image.new("RGB", (canvas_w, canvas_h), "white")
    draw = ImageDraw.Draw(image)
    line_width = max(1, int(round(stroke_width * full_scale)))
    dot_radius = max(1.0, stroke_width * full_scale / 2.0)

    for trace in sample.traces:
        points = [
            ((x - min_x + margin) * full_scale, (y - min_y + margin) * full_scale)
            for x, y in trace
        ]
        if len(points) == 1:
            x, y = points[0]
            draw.ellipse((x - dot_radius, y - dot_radius, x + dot_radius, y + dot_radius), fill="black")
        else:
            draw.line(points, fill="black", width=line_width, joint="curve")

    if aa > 1:
        final_w = max(1, int(math.ceil(width_units * draw_scale)))
        final_h = max(1, int(math.ceil(height_units * draw_scale)))
        image = image.resize((final_w, final_h), Image.Resampling.LANCZOS)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def relative_to(path: Path, root: Path) -> str:
    try:
        return os.path.relpath(path.resolve(), root.resolve())
    except ValueError:
        return str(path.resolve())


def render_and_build_row(payload: tuple[InkSample, str, str, argparse.Namespace]) -> dict[str, str]:
    sample, image_dir_str, eval_root_str, args = payload
    image_dir = Path(image_dir_str)
    eval_root = Path(eval_root_str)
    image_path = image_dir / sample.split / f"{sample.sample_id}.png"
    if args.force or not image_path.exists():
        render_sample(
            sample,
            image_path,
            stroke_width=args.stroke_width,
            margin=args.margin,
            scale=args.scale,
            antialias=args.antialias,
            max_render_side=args.max_render_side,
        )
    rel_image_path = relative_to(image_path, eval_root)
    return {
        "sample_id": sample.sample_id,
        "split": sample.split,
        "crop_image": rel_image_path,
        "image_path": rel_image_path,
        "reference": sample.label,
        "inkml_path": relative_to(sample.inkml_path, eval_root),
    }


def main() -> None:
    args = parse_args()
    eval_root = Path.cwd().resolve()
    data_root = maybe_dataset_root(args.data_root)
    image_dir = args.image_dir.resolve()
    manifest_path = args.manifest.resolve()

    samples = list(iter_samples(data_root, args.splits))
    if args.limit:
        samples = samples[: args.limit]

    payloads = [(sample, str(image_dir), str(eval_root), args) for sample in samples]
    if args.num_workers > 1:
        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            rows = list(tqdm(executor.map(render_and_build_row, payloads), total=len(payloads), desc="render"))
    else:
        rows = [render_and_build_row(payload) for payload in tqdm(payloads, desc="render")]

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "data_root": str(data_root),
        "manifest": str(manifest_path),
        "image_dir": str(image_dir),
        "splits": args.splits,
        "samples": len(rows),
    }
    summary_path = manifest_path.with_suffix(".summary.json")
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
