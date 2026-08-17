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

COMMON_SETS = ('IIIT5k', 'SVT', 'SVTP', 'IC13_857', 'IC15_1811', 'CUTE80')
U14M_SETS = ('artistic', 'contextless', 'curve', 'general', 'multi_oriented', 'multi_words', 'salient')
OST_SETS = ('heavy', 'weak')
COMMON_LMDB_ROOT = ROOT / 'data' / 'english' / 'test'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint')
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--refine_iters', type=int, default=2)
    parser.add_argument('--groups', default='common,u14m,ost')
    parser.add_argument('--json_out', default=None)
    return parser.parse_args()


def move_to_device(images, device):
    if isinstance(images, dict):
        return {key: value.to(device, non_blocking=True) for key, value in images.items()}
    return images.to(device, non_blocking=True)


def eval_dataset(model, dataset, batch_size, num_workers, collate_fn, device, name):
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    total = 0
    correct = 0
    ned = 0
    with torch.inference_mode():
        for images, labels in tqdm(loader, desc=name):
            images = move_to_device(images, device)
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=device.startswith('cuda')):
                output = model.test_step((images, labels), -1)['output']
            total += output.num_samples
            correct += output.correct
            ned += output.ned
    return {
        'accuracy': round(100.0 * correct / total, 4),
        'ned': round(100.0 * (1.0 - ned / total), 4),
        'num_samples': total,
    }


def main():
    args = parse_args()
    groups = {group.strip() for group in args.groups.split(',') if group.strip()}
    u14m_root = ROOT / 'data' / 'english' / 'test' / 'u14m'
    ost_root = ROOT / 'data' / 'english' / 'test' / 'ost'

    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_absolute():
        checkpoint = (ROOT / checkpoint).resolve()

    model = load_from_checkpoint(str(checkpoint), refine_iters=args.refine_iters).eval().to(args.device)
    hp = model.hparams
    transform = SceneTextDataModule.get_transform(tuple(hp.img_size), multi_scales=getattr(hp, 'multi_scales', None))
    uses_monkey = tuple(hp.img_size) in SceneTextDataModule.MONKEY_SIZES or getattr(hp, 'multi_scales', None) is not None
    collate_fn = SceneTextDataModule.monkey_collate_fn if uses_monkey else None

    results = {}
    if 'common' in groups:
        results['common'] = {}
        for name in COMMON_SETS:
            dataset = LmdbDataset(str(COMMON_LMDB_ROOT / name), hp.charset_test, hp.max_label_length, 0, True, True, transform=transform)
            results['common'][name] = eval_dataset(model, dataset, args.batch_size, args.num_workers, collate_fn, args.device, name)
        results['common_simple_mean'] = round(mean(item['accuracy'] for item in results['common'].values()), 4)

    if 'u14m' in groups:
        results['u14m'] = {}
        for name in U14M_SETS:
            dataset = LmdbDataset(str(u14m_root / name), hp.charset_test, hp.max_label_length, 0, True, True, transform=transform)
            results['u14m'][name] = eval_dataset(model, dataset, args.batch_size, args.num_workers, collate_fn, args.device, name)
        results['u14m_simple_mean'] = round(mean(item['accuracy'] for item in results['u14m'].values()), 4)

    if 'ost' in groups:
        results['ost'] = {}
        for name in OST_SETS:
            dataset = LmdbDataset(str(ost_root / name), hp.charset_test, hp.max_label_length, 0, True, True, transform=transform)
            results['ost'][name] = eval_dataset(model, dataset, args.batch_size, args.num_workers, collate_fn, args.device, name)
        results['ost_simple_mean'] = round(mean(item['accuracy'] for item in results['ost'].values()), 4)

    payload = {
        'checkpoint': args.checkpoint,
        'refine_iters': args.refine_iters,
        'batch_size': args.batch_size,
        'num_workers': args.num_workers,
        **results,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
