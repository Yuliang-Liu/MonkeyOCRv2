import importlib.util
import os
import sys
import types
from typing import Sequence

import torch
import torch.nn as nn
from mmengine.dist import is_main_process
from mmengine.logging import MMLogger
from mmengine.model import BaseModule

from mmseg.registry import MODELS
from ..utils import resize
from .timm_dct import FPH, SCSEModule


def _resolve_vitae_attn_implementation(requested: str = None) -> str:
    if requested is None:
        return requested

    if requested != 'sdpa':
        return requested

    if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
        return requested

    return 'eager'


def _build_minimal_vitae_config_module(module_name: str):
    """Build a minimal config module when processor-side imports are unavailable.

    The upstream config file also registers an AutoProcessor and imports
    Qwen2_5_VLProcessor. That is unnecessary for FFDN, which only needs the
    vision encoder config/model classes. This fallback keeps backbone loading
    working on older transformers installs.
    """
    from transformers.configuration_utils import PretrainedConfig

    module = types.ModuleType(module_name)

    class MonkeyOCRv2ViTAEEncoderConfig(PretrainedConfig):
        model_type = 'monkeyocrv2_vitae_encoder'

        def __init__(
            self,
            num_channels: int = 3,
            patch_size: int = 32,
            temporal_patch_size: int = 1,
            stage_dims=None,
            stage_depths=None,
            stage_heads=None,
            downsample_ratios=None,
            kernel_sizes=None,
            rc_tokens_type=None,
            nc_tokens_type=None,
            nc_groups=None,
            rc_groups=None,
            rc_heads=None,
            rc_embed_dims=None,
            prm_embed_dim: int = 64,
            window_size: int = 7,
            mlp_ratio: float = 4.0,
            hidden_size: int = 1024,
            rms_norm_eps: float = 1e-5,
            use_bias: bool = False,
            attn_implementation: str = 'sdpa',
            initializer_range: float = 0.02,
            init_merger_std: float = 0.02,
            is_causal: bool = False,
            post_norm: bool = True,
            gradient_checkpointing: bool = False,
            **kwargs,
        ):
            super().__init__(**kwargs)
            self.num_channels = num_channels
            self.patch_size = patch_size
            self.temporal_patch_size = temporal_patch_size
            self.stage_dims = stage_dims if stage_dims is not None else [64, 128, 256, 512]
            self.stage_depths = stage_depths if stage_depths is not None else [2, 2, 8, 2]
            self.stage_heads = stage_heads if stage_heads is not None else [1, 2, 4, 8]
            self.downsample_ratios = (
                downsample_ratios if downsample_ratios is not None else [4, 2, 2, 2]
            )
            self.kernel_sizes = kernel_sizes if kernel_sizes is not None else [7, 3, 3, 3]
            self.rc_tokens_type = (
                rc_tokens_type
                if rc_tokens_type is not None else
                ['window', 'window', 'transformer', 'transformer']
            )
            self.nc_tokens_type = (
                nc_tokens_type
                if nc_tokens_type is not None else
                ['window', 'window', 'transformer', 'transformer']
            )
            self.nc_groups = nc_groups if nc_groups is not None else [1, 32, 64, 128]
            self.rc_groups = rc_groups if rc_groups is not None else [1, 16, 32, 64]
            self.rc_heads = rc_heads if rc_heads is not None else [1, 1, 2, 4]
            self.rc_embed_dims = rc_embed_dims if rc_embed_dims is not None else [64, 64, 128, 256]
            self.prm_embed_dim = prm_embed_dim
            self.window_size = window_size
            self.mlp_ratio = mlp_ratio
            self.hidden_size = hidden_size
            self.rms_norm_eps = rms_norm_eps
            self.use_bias = use_bias
            self.attn_implementation = attn_implementation
            self.initializer_range = initializer_range
            self.init_merger_std = init_merger_std
            self.is_causal = is_causal
            self.post_norm = post_norm
            self.gradient_checkpointing = gradient_checkpointing

    module.MonkeyOCRv2ViTAEEncoderConfig = MonkeyOCRv2ViTAEEncoderConfig
    return module


def _load_monkeyocrv2_vitae_modules(model_dir: str):
    """Load the external MonkeyOCRv2 ViTAE model files as a package."""
    try:
        import transformers  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            'transformers is required to use MonkeyOCRv2ViTAEDct. '
            'Install it before constructing this backbone.'
        ) from exc

    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f'MonkeyOCRv2 ViTAE model directory not found: {model_dir}')

    package_name = '_ffdn_monkeyocrv2_vitae'
    package = sys.modules.get(package_name)
    if package is None:
        package = types.ModuleType(package_name)
        package.__path__ = [model_dir]
        sys.modules[package_name] = package

    def _load(module_file: str):
        module_name = f'{package_name}.{module_file[:-3]}'
        if module_name in sys.modules:
            return sys.modules[module_name]

        file_path = os.path.join(model_dir, module_file)
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if spec is None or spec.loader is None:
            raise ImportError(f'Unable to load module from {file_path}')
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module

    try:
        config_module = _load('configuration_monkeyocrv2_vitae.py')
    except ImportError as exc:
        # The upstream config file imports processor-side classes that are not
        # needed for backbone-only use. Fall back to a minimal config module if
        # the current transformers build does not expose them.
        if 'Qwen2_5_VLProcessor' not in str(exc):
            raise
        module_name = f'{package_name}.configuration_monkeyocrv2_vitae'
        config_module = _build_minimal_vitae_config_module(module_name)
        sys.modules[module_name] = config_module
    modeling_module = _load('modeling_monkeyocrv2_vitae_vision.py')
    return (
        config_module.MonkeyOCRv2ViTAEEncoderConfig,
        modeling_module.MonkeyOCRv2ViTAEVisionTransformer,
    )


def _load_vitae_state_dict(model_dir: str):
    safetensors_path = os.path.join(model_dir, 'model.safetensors')
    pytorch_path = os.path.join(model_dir, 'pytorch_model.bin')
    if os.path.isfile(safetensors_path):
        try:
            from safetensors.torch import load_file
        except ImportError as exc:
            raise ImportError(
                'Loading MonkeyOCRv2 ViTAE from model.safetensors requires safetensors.'
            ) from exc
        return load_file(safetensors_path)
    if os.path.isfile(pytorch_path):
        return torch.load(pytorch_path, map_location='cpu')
    raise FileNotFoundError(
        'No supported MonkeyOCRv2 ViTAE weight file found. Expected '
        f'`model.safetensors` or `pytorch_model.bin` in {model_dir}.'
    )


def _get_vitae_weight_file(model_dir: str) -> str:
    safetensors_path = os.path.join(model_dir, 'model.safetensors')
    pytorch_path = os.path.join(model_dir, 'pytorch_model.bin')
    if os.path.isfile(safetensors_path):
        return safetensors_path
    if os.path.isfile(pytorch_path):
        return pytorch_path
    return '<unknown>'


def _summarize_tensor(tensor: torch.Tensor):
    tensor = tensor.detach().float().cpu()
    return dict(
        shape=tuple(tensor.shape),
        mean=tensor.mean().item(),
        std=tensor.std(unbiased=False).item(),
        norm=tensor.norm().item(),
    )


def _attach_vitae_load_info(model, model_dir: str, load_method: str):
    sample_tensors = {}
    named_params = dict(model.named_parameters())
    for key in (
        'stem.prm_proj.weight',
        'stage1.0.qkv.weight',
        'stage4.0.qkv.weight',
    ):
        if key in named_params:
            sample_tensors[key] = _summarize_tensor(named_params[key])

    model._ffdn_vitae_load_info = dict(
        model_dir=model_dir,
        weight_file=_get_vitae_weight_file(model_dir),
        load_method=load_method,
        stage_dims=list(getattr(model.config, 'stage_dims', [])),
        stage_depths=list(getattr(model.config, 'stage_depths', [])),
        sample_tensors=sample_tensors,
    )
    return model


def _build_monkeyocrv2_vitae_model(model_dir: str, config, model_cls, local_files_only: bool):
    try:
        model = model_cls.from_pretrained(
            model_dir,
            config=config,
            local_files_only=local_files_only,
        )
        return _attach_vitae_load_info(model, model_dir, load_method='from_pretrained')
    except AttributeError as exc:
        message = str(exc)
        supported_messages = (
            "'NoneType' object has no attribute 'get'",
            "object has no attribute 'all_tied_weights_keys'",
        )
        if not any(token in message for token in supported_messages):
            raise

    model = model_cls(config)
    state_dict = _load_vitae_state_dict(model_dir)
    incompatible = model.load_state_dict(state_dict, strict=False)
    missing_keys = list(incompatible.missing_keys)
    unexpected_keys = list(incompatible.unexpected_keys)
    if missing_keys or unexpected_keys:
        raise RuntimeError(
            'Failed to load MonkeyOCRv2 ViTAE weights cleanly. '
            f'Missing keys: {missing_keys[:20]}; '
            f'Unexpected keys: {unexpected_keys[:20]}'
        )
    return _attach_vitae_load_info(model, model_dir, load_method='manual_state_dict')


def _log_load_info(model, model_dir: str, prefix: str):
    if not is_main_process():
        return
    logger = MMLogger.get_current_instance()
    info = getattr(model, '_ffdn_vitae_load_info', None)
    if info is None:
        stage_dims = getattr(model.config, 'stage_dims', None)
        stage_depths = getattr(model.config, 'stage_depths', None)
        message = '\n'.join([
            f'{prefix}: MonkeyOCRv2 ViTAE pretrained backbone loaded.',
            f'  model_dir={model_dir}',
            f'  stage_dims={stage_dims}',
            f'  stage_depths={stage_depths}',
        ])
    else:
        lines = [
            f'{prefix}: MonkeyOCRv2 ViTAE pretrained weights loaded successfully.',
            f'  load_method={info["load_method"]}',
            f'  model_dir={info["model_dir"]}',
            f'  weight_file={info["weight_file"]}',
            f'  stage_dims={info["stage_dims"]}',
            f'  stage_depths={info["stage_depths"]}',
        ]
        for key, stats in info['sample_tensors'].items():
            lines.append(
                '  '
                f'{key}: shape={stats["shape"]}, '
                f'mean={stats["mean"]:.6f}, std={stats["std"]:.6f}, norm={stats["norm"]:.6f}'
            )
        message = '\n'.join(lines)
    if logger is not None:
        logger.info(message)
    else:
        print(message)


@MODELS.register_module()
class MonkeyOCRv2ViTAEDct(BaseModule):
    """MonkeyOCRv2 ViTAEv2-S encoder + original FFDN DCT fusion.

    The external MonkeyOCRv2 ViTAE encoder returns four packed token stages at
    strides 4/8/16/32. This wrapper reconstructs them into feature maps and
    injects JPEG DCT cues at the stride-8 stage with FFDN's zero-initialized
    residual fusion.
    """

    def __init__(
        self,
        pretrained_model_name_or_path: str,
        in_channels: int = 3,
        out_indices: Sequence[int] = (0, 1, 2, 3),
        fusion: str = 'ZERO',
        local_files_only: bool = True,
        freeze_rgb_encoder: bool = False,
        use_bf16: bool = False,
        attn_implementation: str = None,
        init_cfg=None,
    ):
        super().__init__(init_cfg=init_cfg)
        if in_channels != 3:
            raise ValueError('MonkeyOCRv2ViTAEDct only supports RGB input with 3 channels')

        config_cls, model_cls = _load_monkeyocrv2_vitae_modules(pretrained_model_name_or_path)
        config = config_cls.from_pretrained(
            pretrained_model_name_or_path,
            local_files_only=local_files_only,
        )
        if attn_implementation is not None:
            config.attn_implementation = _resolve_vitae_attn_implementation(
                attn_implementation
            )

        self.rgb_encoder = _build_monkeyocrv2_vitae_model(
            pretrained_model_name_or_path,
            config=config,
            model_cls=model_cls,
            local_files_only=local_files_only,
        )
        _log_load_info(self.rgb_encoder, pretrained_model_name_or_path, 'MonkeyOCRv2ViTAEDct')

        self.out_indices = tuple(out_indices)
        self.stage_dims = tuple(config.stage_dims)
        self.patch_size = config.patch_size
        self.temporal_patch_size = config.temporal_patch_size
        self.use_bf16 = use_bf16
        if freeze_rgb_encoder:
            self.rgb_encoder.requires_grad_(False)

        self.fph = FPH()
        self.fusion = fusion
        if fusion != 'ZERO':
            raise NotImplementedError('MonkeyOCRv2ViTAEDct currently keeps the original ZERO fusion only.')

        if len(self.out_indices) < 2:
            raise ValueError('out_indices must include the stride-8 stage for DCT fusion.')
        self.fusion_stage_idx = self.out_indices[1]
        self.fusion_channels = self.stage_dims[self.fusion_stage_idx]

        self.f_dct_proj = nn.Sequential(
            nn.Conv2d(256, self.fusion_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.fusion_channels),
            nn.ReLU(True),
        )
        self.fusion_pre = nn.Sequential(
            SCSEModule(self.fusion_channels * 2),
            nn.Conv2d(
                self.fusion_channels * 2,
                self.fusion_channels,
                kernel_size=3,
                stride=1,
                padding=1),
            nn.BatchNorm2d(self.fusion_channels),
            nn.ReLU(True),
        )
        self.fusion_post = nn.Conv2d(
            self.fusion_channels,
            self.fusion_channels,
            kernel_size=1,
            stride=1,
            padding=0)
        nn.init.zeros_(self.fusion_post.weight)
        if self.fusion_post.bias is not None:
            nn.init.zeros_(self.fusion_post.bias)

    def _tileify(self, x: torch.Tensor):
        b, c, h, w = x.shape
        if h % self.patch_size != 0 or w % self.patch_size != 0:
            raise ValueError(
                f'Input size {(h, w)} must be divisible by MonkeyOCRv2 ViTAE '
                f'patch_size {self.patch_size}.'
            )

        grid_h = h // self.patch_size
        grid_w = w // self.patch_size
        tiles = x.reshape(
            b,
            c,
            grid_h,
            self.patch_size,
            grid_w,
            self.patch_size,
        )
        tiles = tiles.permute(0, 2, 4, 1, 3, 5).reshape(
            b * grid_h * grid_w,
            c,
            self.patch_size,
            self.patch_size,
        ).contiguous()
        grid_thw = torch.tensor(
            [[1, grid_h, grid_w]] * b,
            device=x.device,
            dtype=torch.long,
        )
        return tiles, grid_thw

    def _packed_stage_to_map(self, tokens: torch.Tensor, grid_hw: torch.Tensor):
        features = []
        offset = 0
        for h, w in grid_hw.tolist():
            h, w = int(h), int(w)
            num_tokens = h * w
            feat = tokens[offset:offset + num_tokens]
            offset += num_tokens
            features.append(feat.transpose(0, 1).reshape(-1, h, w))
        return torch.stack(features, dim=0).contiguous()

    def _encode_rgb(self, x: torch.Tensor):
        tiles, grid_thw = self._tileify(x)
        if self.use_bf16:
            tiles = tiles.bfloat16()
        hidden_states_all, grid_hw_all = self.rgb_encoder(tiles, grid_thw)
        outs = []
        for stage_idx in self.out_indices:
            feat = self._packed_stage_to_map(
                hidden_states_all[stage_idx].float(),
                grid_hw_all[stage_idx],
            )
            outs.append(feat)
        return outs

    def forward(self, inputs):
        x, dct, qtb = inputs['x'], inputs['dct'], inputs['qtb']
        outs = self._encode_rgb(x)

        fusion_out_idx = self.out_indices.index(self.fusion_stage_idx)
        stage_feat = outs[fusion_out_idx]
        f_dct = self.fph(dct, qtb).float()
        f_dct = self.f_dct_proj(f_dct)
        if f_dct.shape[2:] != stage_feat.shape[2:]:
            f_dct = resize(
                f_dct,
                size=stage_feat.shape[2:],
                mode='bilinear',
                align_corners=False)

        ext = self.fusion_pre(torch.cat((stage_feat, f_dct), dim=1))
        outs[fusion_out_idx] = self.fusion_post(ext) + stage_feat
        return tuple(outs)
