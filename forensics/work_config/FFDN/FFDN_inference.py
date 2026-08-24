import os

_base_ = [
    '../_base_/default_runtime.py',
]

num_classes = 2
size = 512


custom_imports = dict(imports=['mmseg.models.backbones.timm_dct',
                               'mmseg.models.necks.dwt'],
                      allow_failed_imports=False)

data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    size=(size, size),
    bgr_to_rgb=True,
    pad_val=255,  # 255 for img
    seg_pad_val=0,  # 0 for mask
    binary_seg=True
)

norm_cfg = dict(type='SyncBN', requires_grad=True)
