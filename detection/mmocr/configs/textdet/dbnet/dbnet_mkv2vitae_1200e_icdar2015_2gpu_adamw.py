_base_ = [
    'dbnet_mkv2vitae_1200e_icdar2015_2gpu.py',
]

load_from = None

# ---------------------------------------------------------------------------
# Backbone: MonkeyOCRv2-ViTAE (4 stages → res2/3/4/5, strides 4/8/16/32)
# ---------------------------------------------------------------------------
# The data preprocessor below must be configured with
# bgr_to_rgb=True, mean=None, std=None and pad_size_divisor=32 so that the
# backbone receives float RGB images whose H/W are multiples of patch_size=32.
_base_.model.backbone = dict(
    type='MonkeyOCRv2ViTAEBackbone',
    model_dir='pretrained/monkeyocrv2_as',
    weights='pretrained/monkeyocrv2_as/model.safetensors',
    out_features=('res2', 'res3', 'res4', 'res5'),
    out_channels=0,           # keep native stage dims [64, 128, 256, 512]
    use_bf16=True,           # cast ViTAE parameters before DDP construction
    trust_remote_code=True,
    freeze=False,
    gradient_checkpointing=False,
)

# ---------------------------------------------------------------------------
# Neck: FPNC now consumes the 4-stage ViTAE features (in_channels matches
# the native ViTAE stage_dims because out_channels=0 in the backbone).
# ---------------------------------------------------------------------------
_base_.model.neck = dict(
    type='FPNC',
    in_channels=[64, 128, 256, 512],
    lateral_channels=256)

# ---------------------------------------------------------------------------
# Data preprocessor: no mean/std normalization, bgr_to_rgb still on, padded
# to a multiple of 32 (the ViTAE patch_size).
# ---------------------------------------------------------------------------
_base_.model.data_preprocessor = dict(
    type='TextDetDataPreprocessor',
    mean=None,
    std=None,
    bgr_to_rgb=True,
    pad_size_divisor=32)

# ---------------------------------------------------------------------------
# Two-GPU training, total batch size = 16 (= 8 per GPU).
# ---------------------------------------------------------------------------
_base_.train_dataloader.batch_size = 8
_base_.train_dataloader.num_workers = 8
_base_.val_dataloader.batch_size = 1
_base_.val_dataloader.num_workers = 4

# ---------------------------------------------------------------------------
# Optimizer: AdamW with lr=1e-4, weight_decay=1e-4. Weights of bias / norm /
# rms / post_trunk_norm / rotary_pos_emb receive zero weight decay, matching
# the DPText-DETR mkv2vitae_align config.
# ---------------------------------------------------------------------------
_base_.optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=1e-4,
        weight_decay=1e-4),
    paramwise_cfg=dict(
        custom_keys={
            'bias': dict(weight_decay=0.0),
            'norm': dict(weight_decay=0.0),
            'rms': dict(weight_decay=0.0),
            'post_trunk_norm': dict(weight_decay=0.0),
            'rotary_pos_emb': dict(weight_decay=0.0),
        }))

# Some backbone parameters may receive no gradient on a given iteration
# (e.g. when bf16 / DDP / dyanmic-graph interactions are involved). Setting
# `find_unused_parameters=True` avoids spurious DDP crashes.
find_unused_parameters = True

randomness = dict(seed=42)

vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend'),
]
_base_.visualizer.vis_backends = vis_backends

param_scheduler = [
    dict(type='LinearLR', end=100, start_factor=0.001),
    dict(type='PolyLR', power=0.9, eta_min=1e-7, begin=100, end=1200),
]
