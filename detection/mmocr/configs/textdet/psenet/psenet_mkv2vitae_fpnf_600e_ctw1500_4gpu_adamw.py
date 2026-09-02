_base_ = [
    'psenet_mkv2vitae_fpnf_600e_ctw1500_2gpu.py',
]

load_from = None

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

# ---------------------------------------------------------------------------
# Four-GPU training, total batch size = 16 (= 4 per GPU). Per-GPU batch size
# is halved relative to the 2-GPU config so the effective batch size stays
# the same; num_workers is kept at 8 to maintain dataloader throughput.
# ---------------------------------------------------------------------------
_base_.train_dataloader.batch_size = 4
_base_.train_dataloader.num_workers = 8
_base_.val_dataloader.batch_size = 1
_base_.val_dataloader.num_workers = 4
