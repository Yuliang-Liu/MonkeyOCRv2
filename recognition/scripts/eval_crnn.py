#!/usr/bin/env python3

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

EXPERIMENTS = (
    {
        'name': 'crnn_en',
        'language': 'english',
        'checkpoint': Path('model_weight/crnn_en.ckpt'),
    },
    {
        'name': 'crnn_zh',
        'language': 'chinese',
        'checkpoint': Path('model_weight/crnn_zh.ckpt'),
    },
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--out-dir', type=Path, default=Path('eval_results/crnn'))
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--refine-iters', type=int, default=2)
    parser.add_argument('--english-groups', default='common,u14m,ost')
    parser.add_argument('--charset-file', default='charset/ppocr_keys_v1.txt')
    parser.add_argument('--python', default=sys.executable)
    return parser.parse_args()


def run(command: list[str]) -> None:
    print(' '.join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    outputs = {}
    for experiment in EXPERIMENTS:
        checkpoint = experiment['checkpoint']
        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)

        json_out = args.out_dir / f"{experiment['name']}_refine{args.refine_iters}.json"
        if experiment['language'] == 'english':
            command = [
                args.python,
                'scripts/eval_english.py',
                str(checkpoint),
                '--refine_iters',
                str(args.refine_iters),
                '--batch_size',
                str(args.batch_size),
                '--num_workers',
                str(args.num_workers),
                '--device',
                args.device,
                '--groups',
                args.english_groups,
                '--json_out',
                str(json_out),
            ]
        else:
            command = [
                args.python,
                'scripts/eval_chinese.py',
                str(checkpoint),
                '--charset_file',
                args.charset_file,
                '--refine_iters',
                str(args.refine_iters),
                '--batch_size',
                str(args.batch_size),
                '--num_workers',
                str(args.num_workers),
                '--device',
                args.device,
                '--json_out',
                str(json_out),
            ]
        run(command)
        outputs[experiment['name']] = json.loads(json_out.read_text(encoding='utf-8'))

    summary_path = args.out_dir / f'summary_refine{args.refine_iters}.json'
    summary_path.write_text(json.dumps(outputs, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f'summary: {summary_path}', flush=True)


if __name__ == '__main__':
    main()
