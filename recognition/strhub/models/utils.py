from pathlib import Path, PurePath
from typing import Sequence

import yaml

import torch
from torch import nn


class InvalidModelError(RuntimeError):
    pass


_WEIGHTS_URL = {
    'parseq': 'https://github.com/baudm/parseq/releases/download/v1.0.0/parseq-bb5792a6.pt',
}


def _get_config(experiment: str, **kwargs):
    root = PurePath(__file__).parents[2]
    with open(root / 'configs/main.yaml', 'r') as f:
        config = yaml.load(f, yaml.Loader)['model']
    with open(root / 'configs/charset/94_full.yaml', 'r') as f:
        config.update(yaml.load(f, yaml.Loader)['model'])
    with open(root / f'configs/experiment/{experiment}.yaml', 'r') as f:
        exp = yaml.load(f, yaml.Loader)
    model = exp['defaults'][0]['override /model']
    with open(root / f'configs/model/{model}.yaml', 'r') as f:
        config.update(yaml.load(f, yaml.Loader))
    if 'model' in exp:
        config.update(exp['model'])
    config.update(kwargs)
    config['lr'] = float(config['lr'])
    return config


def _get_model_class(key):
    if 'crnn' in key:
        from .crnn.system import CRNN as ModelClass
    elif 'parseq' in key:
        from .parseq.system import PARSeq as ModelClass
    else:
        raise InvalidModelError(f"Unable to find model class for '{key}'")
    return ModelClass


def get_pretrained_weights(experiment):
    try:
        url = _WEIGHTS_URL[experiment]
    except KeyError:
        raise InvalidModelError(f"No pretrained weights found for '{experiment}'") from None
    return torch.hub.load_state_dict_from_url(url=url, map_location='cpu', check_hash=True)


def create_model(experiment: str, pretrained: bool = False, **kwargs):
    try:
        config = _get_config(experiment, **kwargs)
    except FileNotFoundError:
        raise InvalidModelError(f"No configuration found for '{experiment}'") from None
    ModelClass = _get_model_class(experiment)
    model = ModelClass(**config)
    if pretrained:
        model.model.load_state_dict(get_pretrained_weights(experiment))
    return model


def load_from_checkpoint(checkpoint_path: str, **kwargs):
    if checkpoint_path.startswith('pretrained='):
        model_id = checkpoint_path.split('=', maxsplit=1)[1]
        model = create_model(model_id, True, **kwargs)
    else:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        hparams = checkpoint.get('hyper_parameters', {})
        try:
            ModelClass = _get_model_class(checkpoint_path)
        except InvalidModelError:
            model_name = hparams.get('name', '')
            if not model_name:
                raise InvalidModelError(
                    f"Unable to determine model class from checkpoint '{checkpoint_path}'"
                ) from None
            ModelClass = _get_model_class(model_name)

        saved_vit_dir = hparams.get('monkey_vit_dir')
        if 'monkey_vit_dir' not in kwargs and saved_vit_dir and not Path(saved_vit_dir).is_dir():
            relative_vit_dir = Path('pretrained/monkeyocr_vit')
            project_root = Path(__file__).resolve().parents[2]
            if (project_root / relative_vit_dir).is_dir():
                kwargs['monkey_vit_dir'] = relative_vit_dir.as_posix()
        model = ModelClass.load_from_checkpoint(checkpoint_path, weights_only=False, **kwargs)
    return model


def parse_model_args(args):
    kwargs = {}
    arg_types = {t.__name__: t for t in [int, float, str]}
    arg_types['bool'] = lambda v: v.lower() == 'true'
    for arg in args:
        name, value = arg.split('=', maxsplit=1)
        name, arg_type = name.split(':', maxsplit=1)
        kwargs[name] = arg_types[arg_type](value)
    return kwargs


def init_weights(module: nn.Module, name: str = '', exclude: Sequence[str] = ()):
    if any(map(name.startswith, exclude)):
        return
    if isinstance(module, nn.Linear):
        nn.init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.trunc_normal_(module.weight, std=0.02)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()
    elif isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.LayerNorm, nn.BatchNorm2d, nn.GroupNorm)):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)
