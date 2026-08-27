_base_ = [
    'psenet_resnet50_fpnf_600e_ctw1500.py',
]

load_from = None

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=20,
        save_best='icdar/hmean',
        rule='greater'))

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
    use_bf16=True,            # cast ViTAE parameters before DDP construction
    trust_remote_code=True,
    freeze=False,
    gradient_checkpointing=False,
)

# ---------------------------------------------------------------------------
# Neck: FPNF now consumes the 4-stage ViTAE features (in_channels matches
# the native ViTAE stage_dims because out_channels=0 in the backbone).
# ---------------------------------------------------------------------------
_base_.model.neck = dict(
    type='FPNF',
    in_channels=[64, 128, 256, 512],
    out_channels=256,
    fusion_type='concat')

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

# Some backbone parameters may receive no gradient on a given iteration
# (e.g. when bf16 / DDP / dynamic-graph interactions are involved). Setting
# `find_unused_parameters=True` avoids spurious DDP crashes.
find_unused_parameters = True

randomness = dict(seed=42)

vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend'),
]
_base_.visualizer.vis_backends = vis_backends
