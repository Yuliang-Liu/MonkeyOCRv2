#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path
from statistics import mean

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from strhub.data.dataset import LmdbDataset
from strhub.data.module import SceneTextDataModule
from strhub.models.utils import load_from_checkpoint


def resolve_charset(charset_file: Path) -> str:
    text = charset_file.read_text(encoding='utf-8', errors='ignore')
    lines = [line.rstrip('\n').rstrip('\r') for line in text.splitlines()]
    return lines[0] if len(lines) == 1 else ''.join(lines)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint')
    parser.add_argument('--charset_file', default=str(ROOT / 'charset' / 'ppocr_keys_v1.txt'))
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--refine_iters', type=int, default=2)
    parser.add_argument('--json_out', default=None)
    args = parser.parse_args()

    charset = resolve_charset(Path(args.charset_file))
    test_root = ROOT / 'data' / 'chinese' / 'test'
    datasets = ('scene_test', 'web_test', 'document_test', 'handwriting_test')

    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_absolute():
        checkpoint = (ROOT / checkpoint).resolve()

    model = load_from_checkpoint(str(checkpoint), charset_test=charset, refine_iters=args.refine_iters).eval().to(args.device)
    hp = model.hparams
    transform = SceneTextDataModule.get_transform(tuple(hp.img_size), multi_scales=getattr(hp, 'multi_scales', None))

    results = {}
    for name in datasets:
        dataset = LmdbDataset(str(test_root / name), charset, hp.max_label_length, 0, True, False, transform=transform)
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=True,
            persistent_workers=args.num_workers > 0,
            collate_fn=SceneTextDataModule.monkey_collate_fn,
        )
        outputs = []
        for images, labels in tqdm(loader, desc=name):
            images = {key: value.to(model.device, non_blocking=True) for key, value in images.items()}
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=args.device.startswith('cuda')):
                output = model.test_step((images, labels), -1)['output']
            outputs.append({'output': output})
        acc, ned, _ = model._aggregate_results(outputs)
        results[name] = {
            'accuracy': round(100 * float(acc), 4),
            'ned': round(100 * float(ned), 4),
            'num_samples': len(dataset),
        }

    payload = {
        'checkpoint': args.checkpoint,
        'refine_iters': args.refine_iters,
        'results': results,
        'simple_mean': round(mean(item['accuracy'] for item in results.values()), 4),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
