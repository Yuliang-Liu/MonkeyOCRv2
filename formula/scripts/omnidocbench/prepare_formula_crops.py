#!/usr/bin/env python3
"""Prepare display-formula crops from OmniDocBench JSON.

The output manifest keeps enough information to merge predictions back into the
original OmniDocBench JSON.  It is intentionally format-tolerant, but the main
target is the official structure:

[
  {
    "layout_dets": [
      {"category_type": "equation_isolated", "poly": [...], "latex": "..."}
    ],
    "page_info": {"image_path": "..."}
  }
]
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Iterable

from PIL import Image
from tqdm import tqdm


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def json_pointer(path: list[Any]) -> str:
    parts = []
    for part in path:
        text = str(part).replace("~", "~0").replace("/", "~1")
        parts.append(text)
    return "/" + "/".join(parts)


def iter_layout_nodes(data: Any) -> Iterable[tuple[list[Any], dict[str, Any], dict[str, Any]]]:
    if isinstance(data, list):
        for page_idx, page in enumerate(data):
            if not isinstance(page, dict):
                continue
            page_info = page.get("page_info", {}) if isinstance(page.get("page_info"), dict) else {}
            layout = page.get("layout_dets", [])
            if not isinstance(layout, list):
                continue
            for det_idx, det in enumerate(layout):
                if isinstance(det, dict):
                    yield [page_idx, "layout_dets", det_idx], det, page_info
    elif isinstance(data, dict):
        page_info = data.get("page_info", {}) if isinstance(data.get("page_info"), dict) else {}
        layout = data.get("layout_dets", [])
        if isinstance(layout, list):
            for det_idx, det in enumerate(layout):
                if isinstance(det, dict):
                    yield ["layout_dets", det_idx], det, page_info


def poly_to_bbox(poly: list[float]) -> tuple[int, int, int, int]:
    xs = [float(x) for x in poly[0::2]]
    ys = [float(y) for y in poly[1::2]]
    left, right = min(xs), max(xs)
    top, bottom = min(ys), max(ys)
    return int(left), int(top), int(right), int(bottom)


def node_bbox(node: dict[str, Any]) -> tuple[int, int, int, int] | None:
    if isinstance(node.get("poly"), list) and len(node["poly"]) >= 8:
        return poly_to_bbox(node["poly"])
    if isinstance(node.get("bbox"), list) and len(node["bbox"]) >= 4:
        x0, y0, x1, y1 = [float(v) for v in node["bbox"][:4]]
        return int(min(x0, x1)), int(min(y0, y1)), int(max(x0, x1)), int(max(y0, y1))
    return None


def resolve_image_path(image_root: Path, image_value: str) -> Path:
    raw = Path(image_value)
    candidates = []
    if raw.is_absolute():
        candidates.append(raw)
        candidates.append(image_root / raw.name)
    else:
        candidates.append(image_root / raw)
        candidates.append(image_root / raw.name)

    for candidate in candidates:
        if candidate.exists():
            return candidate

    # Fallback: basename search in the image root.  This keeps the script usable
    # when OmniDocBench stores paths with a different parent directory.
    basename = raw.name
    for match in image_root.rglob(basename):
        if match.suffix.lower() in IMAGE_EXTS:
            return match
    return candidates[0]


def get_page_image(page_info: dict[str, Any]) -> str | None:
    for key in ("image_path", "img_path", "image", "filename", "file_name"):
        value = page_info.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation", required=True, type=Path)
    parser.add_argument("--image-root", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--category", default="equation_isolated")
    parser.add_argument("--gt-key", default="latex")
    parser.add_argument("--margin", type=int, default=2)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = load_json(args.annotation)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for path, node, page_info in iter_layout_nodes(data):
        if node.get("category_type") != args.category:
            continue
        if node.get("ignore") is True:
            continue
        page_image = get_page_image(page_info)
        bbox = node_bbox(node)
        if not page_image or bbox is None:
            rows.append(
                {
                    "sample_id": f"{len(rows):08d}",
                    "json_pointer": json_pointer(path),
                    "missing_reason": "missing_page_image_or_bbox",
                    "reference": node.get(args.gt_key, ""),
                }
            )
            continue
        sample_id = f"{len(rows):08d}"
        image_path = resolve_image_path(args.image_root, page_image)
        crop_name = f"{sample_id}.png"
        crop_path = args.out_dir / crop_name
        row = {
            "sample_id": sample_id,
            "json_pointer": json_pointer(path),
            "page_image": str(image_path),
            "crop_image": str(crop_path),
            "reference": node.get(args.gt_key, ""),
            "category_type": node.get("category_type"),
            "anno_id": node.get("anno_id"),
            "poly": node.get("poly"),
            "bbox": list(bbox),
        }
        rows.append(row)
        if args.limit and len(rows) >= args.limit:
            break

    if args.dry_run:
        print(json.dumps({"num_formula_nodes": len(rows)}, indent=2))
        return

    written = 0
    for row in tqdm(rows, desc="crop"):
        if "missing_reason" in row:
            continue
        image_path = Path(row["page_image"])
        if not image_path.exists():
            row["missing_reason"] = "page_image_not_found"
            continue
        with Image.open(image_path) as img:
            left, top, right, bottom = row["bbox"]
            left = max(0, left - args.margin)
            top = max(0, top - args.margin)
            right = min(img.width, right + args.margin)
            bottom = min(img.height, bottom + args.margin)
            if right <= left or bottom <= top:
                row["missing_reason"] = "empty_bbox"
                continue
            img.crop((left, top, right, bottom)).convert("RGB").save(row["crop_image"])
            written += 1

    with args.manifest.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(json.dumps({"manifest": str(args.manifest), "rows": len(rows), "crops": written}, indent=2))


if __name__ == "__main__":
    main()

