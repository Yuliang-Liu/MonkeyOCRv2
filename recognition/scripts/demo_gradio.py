#!/usr/bin/env python3
import sys
from functools import lru_cache
from pathlib import Path

import gradio as gr
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from strhub.data.module import SceneTextDataModule
from strhub.models.utils import load_from_checkpoint

MODEL_PATHS = {
    'PARSeq English': 'model_weight/parseq_en.ckpt',
    'PARSeq Chinese': 'model_weight/parseq_zh.ckpt',
    'CRNN English': 'model_weight/crnn_en.ckpt',
    'CRNN Chinese': 'model_weight/crnn_zh.ckpt',
}


def resolve_charset(charset_file: Path) -> str:
    text = charset_file.read_text(encoding='utf-8', errors='ignore')
    lines = [line.rstrip('\n').rstrip('\r') for line in text.splitlines()]
    return lines[0] if len(lines) == 1 else ''.join(lines)


@lru_cache(maxsize=8)
def get_model(model_name: str, device: str):
    checkpoint = MODEL_PATHS[model_name]
    kwargs = {'refine_iters': 2}
    if checkpoint.endswith('_zh.ckpt'):
        kwargs['charset_test'] = resolve_charset(ROOT / 'charset' / 'ppocr_keys_v1.txt')
    model = load_from_checkpoint(str(ROOT / checkpoint), **kwargs).eval().to(device)
    return model


@torch.inference_mode()
def predict(image, model_name, device):
    if image is None:
        return ''
    model = get_model(model_name, device)
    transform = SceneTextDataModule.get_transform(tuple(model.hparams.img_size), multi_scales=getattr(model.hparams, 'multi_scales', None))
    pil = image if isinstance(image, Image.Image) else Image.fromarray(image)
    sample = transform(pil.convert('RGB'))
    if isinstance(sample, dict):
        sample = {k: v.to(device) for k, v in sample.items()}
    else:
        sample = sample.unsqueeze(0).to(device)
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=device.startswith('cuda')):
        probs = model(sample).softmax(-1)
    pred, _ = model.tokenizer.decode(probs)
    pred = pred[0]
    return pred


def build_demo():
    return gr.Interface(
        fn=predict,
        inputs=[
            gr.Image(type='pil', label='Image'),
            gr.Radio(
                choices=list(MODEL_PATHS),
                value='PARSeq English',
                label='Model',
            ),
            gr.Dropdown(['cuda', 'cpu'], value='cuda', label='Device'),
        ],
        outputs=gr.Textbox(label='Prediction'),
        title='MonkeyOCRv2 Recognition Demo',
        examples=[
            ['demo_images/ic13_word_256.png', 'PARSeq English', 'cuda'],
            ['demo_images/chinese_scene_test_000001.png', 'PARSeq Chinese', 'cuda'],
            ['demo_images/chinese_scene_test_000001.png', 'CRNN Chinese', 'cuda'],
        ],
    )


def main():
    demo = build_demo()
    demo.launch(server_name='0.0.0.0', server_port=7860)


if __name__ == '__main__':
    main()
