_base_ = [
    '_base_dbnet_resnet50-dcnv2_fpnc.py',
    '../_base_/datasets/totaltext.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_sgd_1200e.py',
]

load_from = None

# ---------------------------------------------------------------------------
# Backbone: ResNet-50 weights from oCLIP (CLIPResNet). Same depth & stage
# dims as the DCNv2 base, so the FPNC neck and DBHead are reused as-is.
# ---------------------------------------------------------------------------
_base_.model.backbone = dict(
    type='CLIPResNet',
    init_cfg=dict(
        type='Pretrained',
        checkpoint='https://download.openmmlab.com/'
        'mmocr/backbone/resnet50-oclip-7ba0c533.pth'))

# ---------------------------------------------------------------------------
# TotalText pipeline: curves & polygons may be invalid, so we add
# ``FixInvalidPolygon`` after loading annotations (mirrors the r18 totaltext
# config).
# ---------------------------------------------------------------------------
train_pipeline = [
    dict(type='LoadImageFromFile', color_type='color_ignore_orientation'),
    dict(
        type='LoadOCRAnnotations',
        with_polygon=True,
        with_bbox=True,
        with_label=True,
    ),
    dict(type='FixInvalidPolygon', min_poly_points=4),
    dict(
        type='TorchVisionWrapper',
        op='ColorJitter',
        brightness=32.0 / 255,
        saturation=0.5),
    dict(
        type='ImgAugWrapper',
        args=[['Fliplr', 0.5],
              dict(cls='Affine', rotate=[-10, 10]), ['Resize', [0.5, 3.0]]]),
    dict(type='RandomCrop', min_side_ratio=0.1),
    dict(type='Resize', scale=(640, 640), keep_ratio=True),
    dict(type='Pad', size=(640, 640)),
    dict(
        type='PackTextDetInputs',
        meta_keys=('img_path', 'ori_shape', 'img_shape'))
]

test_pipeline = [
    dict(type='LoadImageFromFile', color_type='color_ignore_orientation'),
    dict(type='Resize', scale=(1333, 736), keep_ratio=True),
    dict(
        type='LoadOCRAnnotations',
        with_polygon=True,
        with_bbox=True,
        with_label=True,
    ),
    dict(type='FixInvalidPolygon', min_poly_points=4),
    dict(
        type='PackTextDetInputs',
        meta_keys=('img_path', 'ori_shape', 'img_shape', 'scale_factor'))
]

# dataset settings
totaltext_textdet_train = _base_.totaltext_textdet_train
totaltext_textdet_train.pipeline = train_pipeline
totaltext_textdet_test = _base_.totaltext_textdet_test
totaltext_textdet_test.pipeline = test_pipeline

# ---------------------------------------------------------------------------
# Two-GPU training, total batch size = 16 (= 8 per GPU).
# ---------------------------------------------------------------------------
train_dataloader = dict(
    batch_size=8,
    num_workers=8,
    pin_memory=True,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=totaltext_textdet_train)

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=totaltext_textdet_test)

test_dataloader = val_dataloader

# ---------------------------------------------------------------------------
# Optimizer & schedule: a small warmup + PolyLR, like the oCLIP DBNet config.
# ---------------------------------------------------------------------------
_base_.optim_wrapper.optimizer.lr = 0.002

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
