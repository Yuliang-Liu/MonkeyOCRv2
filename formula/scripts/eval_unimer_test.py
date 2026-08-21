import argparse
import json
import os
import random
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from PIL import Image
from rapidfuzz.distance import Levenshtein
from tabulate import tabulate
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import unimernet.tasks as tasks  # noqa: E402
from unimernet.common.config import Config  # noqa: E402
from unimernet.common.metrics import compute_corpus_bleu  # noqa: E402
from unimernet.datasets.builders import *  # noqa: F401,F403,E402
from unimernet.models import *  # noqa: F401,F403,E402
from unimernet.processors import *  # noqa: F401,F403,E402
from unimernet.processors import load_processor  # noqa: E402
from unimernet.tasks import *  # noqa: F401,F403,E402


class MathDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        raw_image = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            image = self.transform(raw_image)
        else:
            image = raw_image
        return image, str(self.image_paths[idx])


def load_data(image_path, math_file):
    image_dir = Path(image_path)
    image_names = sorted([p.name for p in image_dir.glob("*.png")])
    image_paths = [image_dir / name for name in image_names]
    indices = [int(Path(name).stem) for name in image_names]

    with open(math_file, "r", encoding="utf-8") as f:
        eqs = f.read().splitlines()

    math_gts = []
    for idx, image_name in zip(indices, image_names):
        if idx >= len(eqs):
            raise ValueError(f"Annotation index {idx} from {image_name} exceeds {math_file}.")
        math_gts.append(eqs[idx].strip())

    if len(image_paths) != len(math_gts):
        raise ValueError(f"The number of images and formulas differs for {image_path}.")

    return image_paths, math_gts


def normalize_text(text):
    text_reg = r"(\\(operatorname|mathrm|text|mathbf)\s?\*? {.*?})"
    letter = "[a-zA-Z]"
    noletter = r"[\W_^\d]"
    names = [x[0].replace(" ", "") for x in re.findall(text_reg, text)]
    text = re.sub(text_reg, lambda match: str(names.pop(0)), text)
    news = text
    while True:
        text = news
        news = re.sub(r"(?!\\ )(%s)\s+?(%s)" % (noletter, noletter), r"\1\2", text)
        news = re.sub(r"(?!\\ )(%s)\s+?(%s)" % (noletter, letter), r"\1\2", news)
        news = re.sub(r"(%s)\s+?(%s)" % (letter, noletter), r"\1\2", news)
        if news == text:
            break
    return text


def score_text(predictions, references):
    lev_dist = [Levenshtein.normalized_distance(p, r) for p, r in zip(predictions, references)]
    return {
        "bleu": compute_corpus_bleu(predictions, references),
        "edit": sum(lev_dist) / len(lev_dist),
    }


def setup_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True


def parse_args():
    parser = argparse.ArgumentParser(description="Generate UniMERNet predictions in CDM input format.")
    parser.add_argument("--cfg-path", required=True)
    parser.add_argument("--exp-name", required=True)
    parser.add_argument("--output-dir", default="eval_results/unimer_test/cdm_inputs")
    parser.add_argument("--data-root", default="./data/UniMER-Test")
    parser.add_argument("--datasets", nargs="+", default=["cpe", "hwe", "sce", "spe"])
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--save-raw", action="store_true", help="Also save raw, unnormalized model outputs.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="Optional Config overrides in key=value format, kept compatible with unimernet.common.config.Config.",
    )
    return parser.parse_args()


def collate_batch(batch):
    images, paths = zip(*batch)
    if images and isinstance(images[0], dict):
        image_batch = {
            "pixel_values": torch.cat([item["pixel_values"] for item in images], dim=0),
            "image_grid_thw": torch.stack([item["image_grid_thw"] for item in images], dim=0),
        }
    else:
        image_batch = torch.stack(images, dim=0)
    return image_batch, list(paths)


def to_device(image_batch, device):
    if isinstance(image_batch, dict):
        return {key: value.to(device, non_blocking=True) for key, value in image_batch.items()}
    return image_batch.to(device, non_blocking=True)


def main():
    args = parse_args()
    setup_seeds()

    cfg = Config(args)
    task = tasks.setup_task(cfg)
    model = task.build_model(cfg)
    requested_device = args.device
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        requested_device = "cpu"
    device = torch.device(requested_device)
    model.to(device)
    model.eval()

    vis_eval_cfg = cfg.config.datasets.formula_rec_eval.vis_processor.eval
    vis_processor = load_processor(vis_eval_cfg.name, vis_eval_cfg)
    batch_size = int(args.batch_size or cfg.run_cfg.batch_size_eval)
    num_workers = int(args.num_workers if args.num_workers is not None else cfg.run_cfg.num_workers)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "exp_name": args.exp_name,
        "cfg_path": args.cfg_path,
        "checkpoint": str(cfg.config.model.get("finetuned", "")),
        "vis_processor": vis_eval_cfg.name,
        "device": str(device),
        "batch_size": batch_size,
        "datasets": {},
    }

    print(f"arch_name: {cfg.config.model.arch}")
    print(f"model_type: {cfg.config.model.model_type}")
    print(f"checkpoint: {summary['checkpoint']}")
    print(f"vis_processor: {vis_eval_cfg.name}")
    print(f"device: {device}")
    print("=" * 100)

    generate_cfg = cfg.run_cfg.get("generate_cfg", {})
    generation_kwargs = {}
    for key in ("temperature", "do_sample", "top_p"):
        if key in generate_cfg:
            generation_kwargs[key] = generate_cfg[key]

    data_root = Path(args.data_root)
    for subset in args.datasets:
        image_path = data_root / subset
        math_file = data_root / f"{subset}.txt"
        image_list, math_gts = load_data(image_path, math_file)
        if args.max_samples is not None:
            image_list = image_list[: args.max_samples]
            math_gts = math_gts[: args.max_samples]
        dataset = MathDataset(image_list, transform=vis_processor)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=collate_batch,
            pin_memory=torch.cuda.is_available(),
        )

        start = time.time()
        raw_preds = []
        seen_paths = []
        for images, paths in tqdm(dataloader, desc=f"{args.exp_name}/{subset}"):
            images = to_device(images, device)
            with torch.no_grad():
                output = model.generate({"image": images}, **generation_kwargs)
            raw_preds.extend(output["pred_str"])
            seen_paths.extend(paths)

        norm_gts = [normalize_text(gt) for gt in math_gts]
        norm_preds = [normalize_text(pred) for pred in raw_preds]
        scores = score_text(norm_preds, norm_gts)

        cdm_items = []
        raw_items = []
        for path, gt, pred, raw_gt, raw_pred in zip(seen_paths, norm_gts, norm_preds, math_gts, raw_preds):
            image_stem = Path(path).stem
            item = {
                "img_id": f"{subset}_{image_stem}",
                "gt": gt,
                "pred": pred,
                "image_path": path,
            }
            cdm_items.append(item)
            if args.save_raw:
                raw_items.append({**item, "raw_gt": raw_gt, "raw_pred": raw_pred})

        cdm_path = output_dir / f"{args.exp_name}__{subset}.json"
        with open(cdm_path, "w", encoding="utf-8") as f:
            json.dump(cdm_items, f, ensure_ascii=False, indent=2)

        if args.save_raw:
            raw_path = output_dir / f"{args.exp_name}__{subset}_raw.json"
            with open(raw_path, "w", encoding="utf-8") as f:
                json.dump(raw_items, f, ensure_ascii=False, indent=2)

        elapsed = time.time() - start
        summary["datasets"][subset] = {
            "num_samples": len(cdm_items),
            "scores": scores,
            "cdm_input": str(cdm_path),
            "elapsed_sec": elapsed,
        }

        print(f"Evaluation Set: {subset}")
        print(f"Samples: {len(cdm_items)}")
        print(f"Inference Time: {elapsed:.3f}s")
        print(tabulate([[scores['bleu'], scores['edit']]], headers=["bleu up", "edit down"]))
        print("=" * 100)

        with open(output_dir / f"{args.exp_name}_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
