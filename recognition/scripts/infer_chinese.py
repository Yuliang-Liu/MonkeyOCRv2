#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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
    parser.add_argument('--images', nargs='+', required=True)
    parser.add_argument('--charset_file', default=str(ROOT / 'charset' / 'ppocr_keys_v1.txt'))
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--refine_iters', type=int, default=2)
    args = parser.parse_args()

    charset = resolve_charset(Path(args.charset_file))
    model = load_from_checkpoint(args.checkpoint, charset_test=charset, refine_iters=args.refine_iters).eval().to(args.device)
    img_transform = SceneTextDataModule.get_transform(
        tuple(model.hparams.img_size),
        multi_scales=getattr(model.hparams, 'multi_scales', None),
    )

    for fname in args.images:
        image = Image.open(fname).convert('RGB')
        image = img_transform(image)
        image = {k: v.to(args.device) for k, v in image.items()}
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=args.device.startswith('cuda')):
            probs = model(image).softmax(-1)
        pred, probs = model.tokenizer.decode(probs)
        print(f'{fname}\t{pred[0]}\tconf={probs[0].prod().item():.6f}')


if __name__ == '__main__':
    main()
