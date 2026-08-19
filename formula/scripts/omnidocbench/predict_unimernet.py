#!/usr/bin/env python3
"""Run UniMERNet predictions for crops listed in a manifest JSONL."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


@dataclass
class ConfigArgs:
    cfg_path: str
    options: list[str] | None


class CropDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]], processor: Any):
        self.rows = rows
        self.processor = processor

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.rows[idx]
        with Image.open(row["crop_image"]) as img:
            image = self.processor(img.convert("RGB"))
        return {
            "sample_id": row["sample_id"],
            "crop_image": row["crop_image"],
            "reference": row.get("reference", ""),
            "image": image,
        }


def setup_seeds(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True


def resolve_manifest_path(value: str, eval_root: Path, invocation_cwd: Path) -> str:
    path = Path(value)
    if path.is_absolute():
        return str(path)
    for root in (eval_root, invocation_cwd):
        candidate = root / path
        if candidate.exists():
            return str(candidate)
    return str(eval_root / path)


def read_manifest(path: Path, eval_root: Path, invocation_cwd: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("crop_image") and not row.get("missing_reason"):
                row["crop_image"] = resolve_manifest_path(row["crop_image"], eval_root, invocation_cwd)
                rows.append(row)
    return rows


def collate_batch(items: list[dict[str, Any]]) -> dict[str, Any]:
    first_image = items[0]["image"]
    if isinstance(first_image, torch.Tensor):
        images = torch.stack([item["image"] for item in items], dim=0)
    elif isinstance(first_image, dict):
        images = {
            "pixel_values": torch.cat([item["image"]["pixel_values"] for item in items], dim=0),
            "image_grid_thw": torch.stack([item["image"]["image_grid_thw"] for item in items], dim=0),
        }
    else:
        raise TypeError("Only tensor or packed dynamic image batches are supported by this predictor.")

    return {
        "sample_id": [item["sample_id"] for item in items],
        "crop_image": [item["crop_image"] for item in items],
        "reference": [item["reference"] for item in items],
        "image": images,
    }


def move_image_to_device(image: Any, device: torch.device) -> Any:
    if isinstance(image, dict):
        return {key: value.to(device) if torch.is_tensor(value) else value for key, value in image.items()}
    return image.to(device)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--cfg-path", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--cfg-options", nargs="*", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    eval_root = Path(__file__).resolve().parents[1]
    invocation_cwd = Path.cwd().resolve()
    repo_root = args.repo_root.resolve()
    sys.path.insert(0, str(repo_root))
    os.chdir(repo_root)

    # Registration imports must happen after repo_root is on sys.path.
    import unimernet.tasks as tasks
    from unimernet.common.config import Config
    from unimernet.processors import load_processor

    for module_name in (
        "unimernet.datasets.builders",
        "unimernet.models",
        "unimernet.processors",
        "unimernet.tasks",
    ):
        importlib.import_module(module_name)

    setup_seeds()
    cfg = Config(ConfigArgs(str(args.cfg_path), args.cfg_options))
    task = tasks.setup_task(cfg)
    model = task.build_model(cfg)
    vis_eval_cfg = cfg.config.datasets.formula_rec_eval.vis_processor.eval
    processor = load_processor(vis_eval_cfg.name, vis_eval_cfg)

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    model.to(device)
    model.eval()

    rows = read_manifest(args.manifest, eval_root, invocation_cwd)
    if args.limit:
        rows = rows[: args.limit]

    dataset = CropDataset(rows, processor)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=collate_batch,
        shuffle=False,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    generate_cfg = cfg.run_cfg.get("generate_cfg", {})
    generation_kwargs = {}
    for key in ("temperature", "do_sample", "top_p"):
        if key in generate_cfg:
            generation_kwargs[key] = generate_cfg[key]

    with args.output.open("w", encoding="utf-8") as f:
        for batch in tqdm(loader, desc="predict"):
            images = move_image_to_device(batch["image"], device)
            with torch.no_grad():
                output = model.generate({"image": images}, **generation_kwargs)
            for sample_id, crop_image, reference, prediction in zip(
                batch["sample_id"],
                batch["crop_image"],
                batch["reference"],
                output["pred_str"],
            ):
                f.write(
                    json.dumps(
                        {
                            "sample_id": sample_id,
                            "crop_image": crop_image,
                            "reference": reference,
                            "prediction": prediction,
                            "cfg_path": str(args.cfg_path),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    print(json.dumps({"output": str(args.output), "predictions": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
