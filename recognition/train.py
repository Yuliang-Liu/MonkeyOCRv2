#!/usr/bin/env python3
# Scene Text Recognition Model Hub
# Copyright 2022 Darwin Bautista
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import math
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, open_dict

import torch

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, StochasticWeightAveraging
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.utilities.model_summary import summarize

from strhub.data.module import SceneTextDataModule
from strhub.models.base import BaseSystem
from strhub.models.utils import get_pretrained_weights


def resolve_charset(charset_or_path: str) -> str:
    path = Path(charset_or_path)
    if not path.exists():
        return charset_or_path
    text = path.read_text(encoding='utf-8', errors='ignore')
    lines = [line.rstrip('\n').rstrip('\r') for line in text.splitlines()]
    return lines[0] if len(lines) == 1 else ''.join(lines)


def load_pretrained_weights(pretrained: str):
    path = Path(pretrained)
    if path.exists():
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        if not isinstance(state_dict, dict):
            raise TypeError(f'Unsupported checkpoint format: {pretrained}')
        return state_dict
    return get_pretrained_weights(pretrained)


def adapt_pretrained_state_dict(state_dict, model):
    model_keys = set(model.state_dict().keys())
    state_keys = set(state_dict.keys())
    if not (state_keys == model_keys or state_keys.issubset(model_keys)):
        stripped = {
            key[len('model.'):] if key.startswith('model.') else key: value
            for key, value in state_dict.items()
        }
        stripped_keys = set(stripped)
        if stripped_keys == model_keys or stripped_keys.issubset(model_keys):
            state_dict = stripped
        else:
            prefixed = {f'model.{key}': value for key, value in state_dict.items()}
            prefixed_keys = set(prefixed)
            if prefixed_keys == model_keys or prefixed_keys.issubset(model_keys):
                state_dict = prefixed

    # Older English CRNN checkpoints contain 30 unused character slots after
    # the CTC blank. Drop them only while initializing the standard 94-class model.
    model_state = model.state_dict()
    for key in ('model.rnn.1.linear.weight', 'model.rnn.1.linear.bias'):
        if key not in state_dict or key not in model_state:
            continue
        source, target = state_dict[key], model_state[key]
        extra = source.shape[0] - target.shape[0]
        if extra == 30 and source.shape[1:] == target.shape[1:]:
            state_dict[key] = torch.cat((source[:1], source[extra + 1:]), dim=0)
    return state_dict


# Copied from OneCycleLR
def _annealing_cos(start, end, pct):
    'Cosine anneal from `start` to `end` as pct goes from 0.0 to 1.0.'
    cos_out = math.cos(math.pi * pct) + 1
    return end + (start - end) / 2.0 * cos_out


def get_swa_lr_factor(warmup_pct, swa_epoch_start, div_factor=25, final_div_factor=1e4) -> float:
    """Get the SWA LR factor for the given `swa_epoch_start`. Assumes OneCycleLR Scheduler."""
    total_steps = 1000  # Can be anything. We use 1000 for convenience.
    start_step = int(total_steps * warmup_pct) - 1
    end_step = total_steps - 1
    step_num = int(total_steps * swa_epoch_start) - 1
    pct = (step_num - start_step) / (end_step - start_step)
    return _annealing_cos(1, 1 / (div_factor * final_div_factor), pct)


@hydra.main(config_path='configs', config_name='main', version_base='1.2')
def main(config: DictConfig):
    pl.seed_everything(config.seed, workers=True)
    trainer_strategy = 'auto'
    with open_dict(config):
        # Hydra keeps the process in the project root; model paths stay portable.
        config.data.root_dir = hydra.utils.to_absolute_path(config.data.root_dir)
        config.model.charset_train = resolve_charset(config.model.charset_train)
        config.model.charset_test = resolve_charset(config.model.charset_test)
        config.data.charset_train = config.model.charset_train
        config.data.charset_test = config.model.charset_test
        # Special handling for GPU-affected config
        gpu = config.trainer.get('accelerator') == 'gpu'
        devices = config.trainer.get('devices', 0)
        if gpu:
            # Use mixed-precision training
            use_bf16 = bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported())
            config.trainer.precision = 'bf16-mixed' if use_bf16 else '16-mixed'
        if gpu and devices > 1:
            # Use DDP with optimizations
            trainer_strategy = DDPStrategy(find_unused_parameters=False, gradient_as_bucket_view=True)
            # Scale steps-based config
            config.trainer.val_check_interval //= devices
            if config.trainer.get('max_steps', -1) > 0:
                config.trainer.max_steps //= devices

    # Special handling for PARseq
    if config.model.get('perm_mirrored', False):
        assert config.model.perm_num % 2 == 0, 'perm_num should be even if perm_mirrored = True'

    model: BaseSystem = hydra.utils.instantiate(config.model)
    # If specified, use pretrained weights to initialize the model
    if config.pretrained is not None:
        m = model.model if config.model._target_.endswith('PARSeq') else model
        state_dict = adapt_pretrained_state_dict(load_pretrained_weights(config.pretrained), m)
        missing, unexpected = m.load_state_dict(state_dict, strict=False)
        if missing:
            print(f'[pretrained] missing keys: {len(missing)}', flush=True)
        if unexpected:
            print(f'[pretrained] unexpected keys: {len(unexpected)}', flush=True)
    print(summarize(model, max_depth=2))

    datamodule: SceneTextDataModule = hydra.utils.instantiate(config.data)

    checkpoint_monitor = config.get('checkpoint_monitor', 'val_accuracy')
    if checkpoint_monitor in (None, 'null', 'None', ''):
        checkpoint = ModelCheckpoint(
            monitor=None,
            save_top_k=0,
            save_last=True,
            filename='{epoch}-{step}',
        )
    else:
        checkpoint = ModelCheckpoint(
            monitor=checkpoint_monitor,
            mode='max',
            save_top_k=3,
            save_last=True,
            filename='{epoch}-{step}-{val_accuracy:.4f}-{val_NED:.4f}',
        )
    swa_epoch_start = 0.75
    swa_lr = config.model.lr * get_swa_lr_factor(config.model.warmup_pct, swa_epoch_start)
    swa = StochasticWeightAveraging(swa_lr, swa_epoch_start)
    cwd = (
        HydraConfig.get().runtime.output_dir
        if config.ckpt_path is None
        else str(Path(config.ckpt_path).parents[1].absolute())
    )
    trainer: Trainer = hydra.utils.instantiate(
        config.trainer,
        logger=TensorBoardLogger(cwd, '', '.'),
        strategy=trainer_strategy,
        enable_model_summary=False,
        callbacks=[checkpoint, swa],
    )
    trainer.fit(model, datamodule=datamodule, ckpt_path=config.ckpt_path)
    if not config.get('skip_final_validate', False):
        print('[final] running validation on last checkpoint', flush=True)
        trainer.validate(model=model, datamodule=datamodule)


if __name__ == '__main__':
    main()
