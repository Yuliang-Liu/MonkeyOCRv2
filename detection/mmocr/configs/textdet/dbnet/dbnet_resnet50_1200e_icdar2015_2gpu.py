_base_ = [
    'dbnet_resnet50_1200e_icdar2015.py',
]

load_from = None

# Two-GPU training, total batch size = 16 (= 8 per GPU).
_base_.train_dataloader.batch_size = 8
_base_.train_dataloader.num_workers = 8
_base_.val_dataloader.batch_size = 1
_base_.val_dataloader.num_workers = 4

_base_.optim_wrapper.optimizer.lr = 0.002

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

param_scheduler = [
    dict(type='LinearLR', end=100, start_factor=0.001),
    dict(type='PolyLR', power=0.9, eta_min=1e-7, begin=100, end=1200),
]
