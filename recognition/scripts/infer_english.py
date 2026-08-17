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


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint')
    parser.add_argument('--images', nargs='+', required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--refine_iters', type=int, default=2)
    args = parser.parse_args()

    model = load_from_checkpoint(args.checkpoint, refine_iters=args.refine_iters).eval().to(args.device)
    img_transform = SceneTextDataModule.get_transform(
        tuple(model.hparams.img_size),
        multi_scales=getattr(model.hparams, 'multi_scales', None),
    )

    for fname in args.images:
        image = Image.open(fname).convert('RGB')
        image = img_transform(image)
        if isinstance(image, dict):
            image = {k: v.to(args.device) for k, v in image.items()}
        else:
            image = image.unsqueeze(0).to(args.device)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=args.device.startswith('cuda')):
            probs = model(image).softmax(-1)
        pred, probs = model.tokenizer.decode(probs)
        print(f'{fname}	{pred[0]}	conf={probs[0].prod().item():.6f}')


if __name__ == '__main__':
    main()
