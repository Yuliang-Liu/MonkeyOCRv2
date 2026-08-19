#!/usr/bin/env python3
"""Run formula recognition on images with the released MonkeyOCRv2-S model."""

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


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


@dataclass
class ConfigArgs:
    cfg_path: str
    options: list[str] | None


class FormulaImageDataset(Dataset):
    def __init__(self, image_paths: list[Path], processor: Any):
        self.image_paths = image_paths
        self.processor = processor

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        image_path = self.image_paths[idx]
        with Image.open(image_path) as image:
            image = self.processor(image.convert("RGB"))
        return {"image_path": str(image_path), "image": image}


def setup_seeds(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True


def import_unimernet_modules() -> None:
    for module_name in (
        "unimernet.datasets.builders",
        "unimernet.models",
        "unimernet.processors",
        "unimernet.tasks",
    ):
        importlib.import_module(module_name)


def collect_images(images: list[Path], image_dir: Path | None) -> list[Path]:
    paths = [path.resolve() for path in images]
    if image_dir is not None:
        paths.extend(
            sorted(
                path.resolve()
                for path in image_dir.rglob("*")
                if path.is_file() and path.suffix.lower() in IMAGE_EXTS
            )
        )
    if not paths:
        raise ValueError("No input images were provided.")
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing input images: {missing[:5]}")
    return paths


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
        "image_path": [item["image_path"] for item in items],
        "image": collate_images([item["image"] for item in items]),
    }


def to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {key: to_device(item, device) for key, item in value.items()}
    return value


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    parser.add_argument("--cfg-path", type=Path, default=repo_root / "configs/eval/monkeyocrv2_s.yaml")
    parser.add_argument("--checkpoint", type=Path, default=repo_root / "model_weight/monkeyocrv2_s_formula.pth")
    parser.add_argument("--images", nargs="*", type=Path, default=[])
    parser.add_argument("--image-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cfg-options", nargs="*", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    sys.path.insert(0, str(repo_root))
    os.chdir(repo_root)
    setup_seeds()

    import_unimernet_modules()
    import unimernet.tasks as tasks
    from unimernet.common.config import Config
    from unimernet.processors import load_processor

    cfg_options = list(args.cfg_options or [])
    cfg_options.append(f"model.finetuned={args.checkpoint}")
    cfg = Config(ConfigArgs(str(args.cfg_path), cfg_options))

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

    image_paths = collect_images(args.images, args.image_dir)
    dataset = FormulaImageDataset(image_paths, processor)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=collate_batch,
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
    )

    generate_cfg = cfg.run_cfg.get("generate_cfg", {})
    generation_kwargs = {}
    for key in ("temperature", "do_sample", "top_p"):
        if key in generate_cfg:
            generation_kwargs[key] = generate_cfg[key]

    rows = []
    for batch in tqdm(loader, desc="infer"):
        images = to_device(batch["image"], device)
        with torch.no_grad():
            output = model.generate({"image": images}, **generation_kwargs)
        for image_path, prediction in zip(batch["image_path"], output["pred_str"]):
            rows.append({"image_path": image_path, "prediction": prediction})

    if args.output is not None:
        output_path = args.output if args.output.is_absolute() else repo_root / args.output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    for row in rows:
        print(json.dumps(row, ensure_ascii=False))


if __name__ == "__main__":
    main()

