import os

_base_ = [
    '../_base_/default_runtime.py',
]

project = 'FFDN'
name = 'FFDN_monkey_vitae'
group = 'FFDN'
notes = ''
work_dir = f'./work_dirs/{project}/{name}'
wandb_dir = os.path.abspath(os.path.join(work_dir, 'wandb'))
os.makedirs(wandb_dir, exist_ok=True)
num_classes = 2
size = 512
iters = 100000
randomness = dict(seed=42)

monkeyocr_vitae_pretrained_model = os.getenv(
    'monkeyocr_vitae_pretrained_model',
    '/data/dlluo/code/ttd/monkeyocr_vitae-small')
# jpeg_process_style = os.getenv('jpeg_process_style', 'single')
jpeg_process_style = 'single'

base_lr = 1e-4
use_monkeyocr_lr_mult = True
monkeyocr_lr_mult = 0.2

custom_imports = dict(
    imports=[
        'mmseg.models.backbones.monkeyocrv2_vitae_dct',
        'mmseg.models.necks.dwt',
        'mmseg.datasets.transforms.doc_tamper_transforms',
    ],
    allow_failed_imports=False)
find_unused_parameters = True

data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[122.7709383, 116.7460125, 104.09373615],
    std=[68.5005327, 66.6321579, 70.32316305],
    size=(size, size),
    bgr_to_rgb=True,
    pad_val=255,
    seg_pad_val=0,
    binary_seg=True
)

norm_cfg = dict(type='SyncBN', requires_grad=True)

model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type='MonkeyOCRv2ViTAEDct',
        pretrained_model_name_or_path=monkeyocr_vitae_pretrained_model,
        in_channels=3,
        out_indices=(0, 1, 2, 3),
        local_files_only=True,
        freeze_rgb_encoder=False,
        use_bf16=False,
        attn_implementation='sdpa',
    ),
    neck=dict(
        type='DWTFPN',
        in_channels=[64, 128, 256, 512],
        out_channels=256,
        num_outs=4),
    decode_head=dict(
        type='FPNHead',
        in_channels=[256, 256, 256, 256],
        in_index=[0, 1, 2, 3],
        feature_strides=[4, 8, 16, 32],
        channels=512,
        dropout_ratio=0.1,
        num_classes=num_classes,
        norm_cfg=norm_cfg,
        align_corners=False,
        sampler=dict(type='OHEMPixelSampler', thresh=0.9, min_kept=100000),
        loss_decode=[
            dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            dict(type='LovaszLoss', loss_weight=1.0, per_image=False, reduction='none'),
        ],
    ),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=256,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=num_classes,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=[
            dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
        ],
    ),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'),
)

albu_train_transforms = [
    dict(type='HorizontalFlip', p=0.5),
    dict(type='ToGray', p=0.5),
    dict(type='OneOf', transforms=[
        dict(type='VerticalFlip', p=0.5),
        dict(type='RandomRotate90', p=0.5),
        dict(type='Transpose', p=0.5),
    ], p=0.5),
]

data_root = './data/DocTamperV1'
train_pipeline = [
    dict(type='LoadImageLabelFromFileLMDB',
         file_client_args={
             'db_path': os.path.join(data_root, 'unzip_files/DocTamperV1-TrainingSet')
         }),
    dict(type='Resize', scale=(size, size), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='Albu', transforms=albu_train_transforms),
    dict(
        type='RandomJpegCompressAndLoadInfo',
        load_info=True,
        return_rgb=True,
        compress_mode=jpeg_process_style),
    dict(type='PackSegInputs', meta_keys=('img_path', 'seg_map_path', 'ori_shape',
                                          'img_shape', 'pad_shape', 'scale_factor', 'flip',
                                          'flip_direction', 'reduce_zero_label', 'dct', 'qtb')),
]

val_pipeline = [
    dict(type='LoadImageLabelFromFileLMDB',
         file_client_args={
             'db_path': os.getenv('val_db_path',
                                  os.path.join(data_root, 'unzip_files/DocTamperV1-TestingSet'))
         }),
    dict(type='Resize', scale=(size, size), keep_ratio=True),
    dict(
        type='RandomJpegCompressAndLoadInfo',
        load_info=True,
        return_rgb=True,
        compress_mode=jpeg_process_style,
        compress_pk=os.getenv(
            'compress_pk_file',
            os.path.join(data_root, 'pks/DocTamperV1-TestingSet_75.pk'))),
    dict(type='PackSegInputs', meta_keys=('img_path', 'seg_map_path', 'ori_shape',
                                          'img_shape', 'pad_shape', 'scale_factor', 'flip',
                                          'flip_direction', 'reduce_zero_label', 'dct', 'qtb')),
]

dataset_type = 'BaseSegDataset'
metainfo = dict(
    task_name="doc_tamper_segmentation",
    classes=["untamper", "tamper"],
    palette=[[0, 0, 0], [255, 255, 255]],
)

train_dataloader = dict(
    batch_size=4 if not os.getenv('DEBUG') else 2,
    num_workers=4 if not os.getenv('DEBUG') else 0,
    persistent_workers=True if not os.getenv('DEBUG') else False,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='processed/train.txt',
        img_suffix=".jpg",
        seg_map_suffix='.png',
        metainfo=metainfo,
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='processed/sampled_val.txt',
        img_suffix=".jpg",
        seg_map_suffix='.png',
        metainfo=metainfo,
        test_mode=True,
        pipeline=val_pipeline))

test_dataloader = val_dataloader

val_evaluator = [
    dict(type='IoUMetric', iou_metrics=['mIoU', 'mFscore']),
    dict(type='DocTamperMetric', num_classes=num_classes)
]
test_evaluator = val_evaluator

optimizer = dict(type='AdamW', lr=base_lr, betas=(0.9, 0.999), weight_decay=0.05)
optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer, clip_grad=None)
if use_monkeyocr_lr_mult:
    optim_wrapper.update(
        paramwise_cfg=dict(
            custom_keys={
                'backbone.rgb_encoder': dict(lr_mult=monkeyocr_lr_mult),
            }))

param_scheduler = [
    dict(
        type='PolyLR',
        eta_min=0.,
        power=0.9,
        begin=0,
        end=iters,
        by_epoch=False)
]

train_cfg = dict(type='IterBasedTrainLoop', max_iters=iters, val_interval=500)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', save_best='mIoU', rule='greater', by_epoch=False, max_keep_ckpts=1,
                    interval=2000),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook'),
)

vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='WandbAutoFinishBackend', init_kwargs=dict(project=project, group=group, name=name, notes=notes,
                                                         resume=True, allow_val_change=True,
                                                         mode='offline',
                                                         dir=wandb_dir)),
] if not os.getenv('DEBUG') else []

visualizer = dict(type='SegLocalVisualizer', vis_backends=vis_backends, name='visualizer')
