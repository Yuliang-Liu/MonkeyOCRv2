#!/usr/bin/env python3
"""Run UniMERNet-style models on a MathWriting manifest."""

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


class MathWritingImageDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]], processor: Any):
        self.rows = rows
        self.processor = processor

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.rows[idx]
        image_path = row.get("crop_image") or row.get("image_path")
        if image_path is None:
            raise KeyError(f"Manifest row {idx} has no crop_image/image_path field")
        with Image.open(image_path) as img:
            image = self.processor(img.convert("RGB"))
        return {
            "sample_id": row["sample_id"],
            "split": row.get("split", ""),
            "crop_image": image_path,
            "reference": row.get("reference", ""),
            "image": image,
        }


def setup_seeds(seed: int = 3) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True


def resolve_path(value: str, eval_root: Path, invocation_cwd: Path) -> str:
    path = Path(value)
    if path.is_absolute():
        return str(path)
    for root in (eval_root, invocation_cwd):
        candidate = root / path
        if candidate.exists():
            return str(candidate.resolve())
    return str((eval_root / path).resolve())


def read_manifest(path: Path, eval_root: Path, invocation_cwd: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            image_key = "crop_image" if row.get("crop_image") else "image_path"
            row[image_key] = resolve_path(row[image_key], eval_root, invocation_cwd)
            rows.append(row)
    return rows


def collate_images(images: list[Any]) -> Any:
    first = images[0]
    if isinstance(first, torch.Tensor):
        return torch.stack(images, dim=0)
    if isinstance(first, dict):
        batch: dict[str, Any] = {}
        for key in first:
            values = [image[key] for image in images]
            if isinstance(values[0], torch.Tensor):
                if key == "pixel_values":
                    batch[key] = torch.cat(values, dim=0)
                else:
                    batch[key] = torch.stack(values, dim=0)
            else:
                batch[key] = values
        return batch
    raise TypeError(f"Unsupported processor output type: {type(first)!r}")


def collate_batch(items: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "sample_id": [item["sample_id"] for item in items],
        "split": [item["split"] for item in items],
        "crop_image": [item["crop_image"] for item in items],
        "reference": [item["reference"] for item in items],
        "image": collate_images([item["image"] for item in items]),
    }


def to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {key: to_device(item, device) for key, item in value.items()}
    return value


def parse_args() -> argparse.Namespace:
    default_repo = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=default_repo)
    parser.add_argument("--cfg-path", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=-1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--cfg-options", nargs="*", default=None)
    return parser.parse_args()


def import_unimernet_modules() -> None:
    for module_name in (
        "unimernet.datasets.builders",
        "unimernet.models",
        "unimernet.processors",
        "unimernet.tasks",
    ):
        importlib.import_module(module_name)


def main() -> None:
    args = parse_args()
    eval_root = Path(__file__).resolve().parents[1]
    invocation_cwd = Path.cwd().resolve()
    repo_root = args.repo_root.resolve()
    sys.path.insert(0, str(repo_root))
    os.chdir(repo_root)

    import_unimernet_modules()
    import unimernet.tasks as tasks
    from unimernet.common.config import Config
    from unimernet.processors import load_processor

    setup_seeds()
    cfg = Config(ConfigArgs(str(args.cfg_path), args.cfg_options))
    task = tasks.setup_task(cfg)
    model = task.build_model(cfg)
    vis_eval_cfg = cfg.config.datasets.formula_rec_eval.vis_processor.eval
    processor = load_processor(vis_eval_cfg.name, vis_eval_cfg)

    requested_device = args.device
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        requested_device = "cpu"
    device = torch.device(requested_device)
    model.to(device)
    model.eval()

    manifest_path = args.manifest if args.manifest.is_absolute() else invocation_cwd / args.manifest
    rows = read_manifest(manifest_path.resolve(), eval_root, invocation_cwd)
    if args.limit:
        rows = rows[: args.limit]

    batch_size = args.batch_size or int(cfg.run_cfg.batch_size_eval)
    num_workers = args.num_workers if args.num_workers >= 0 else int(cfg.run_cfg.num_workers)
    dataset = MathWritingImageDataset(rows, processor)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collate_batch,
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
    )

    generate_cfg = cfg.run_cfg.get("generate_cfg", {})
    generation_kwargs = {}
    for key in ("temperature", "do_sample", "top_p"):
        if key in generate_cfg:
            generation_kwargs[key] = generate_cfg[key]

    output_path = args.output if args.output.is_absolute() else invocation_cwd / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for batch in tqdm(loader, desc="predict"):
            images = to_device(batch["image"], device)
            with torch.no_grad():
                output = model.generate({"image": images}, **generation_kwargs)
            for sample_id, split, crop_image, reference, prediction in zip(
                batch["sample_id"],
                batch["split"],
                batch["crop_image"],
                batch["reference"],
                output["pred_str"],
            ):
                f.write(
                    json.dumps(
                        {
                            "sample_id": sample_id,
                            "split": split,
                            "crop_image": crop_image,
                            "reference": reference,
                            "prediction": prediction,
                            "cfg_path": str(args.cfg_path),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    print(
        json.dumps(
            {
                "output": str(output_path),
                "predictions": len(rows),
                "cfg_path": str(args.cfg_path),
                "device": str(device),
                "batch_size": batch_size,
                "num_workers": num_workers,
                "checkpoint": str(cfg.config.model.get("finetuned", "")),
                "vis_processor": str(vis_eval_cfg.name),
                "generation_kwargs": generation_kwargs,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
