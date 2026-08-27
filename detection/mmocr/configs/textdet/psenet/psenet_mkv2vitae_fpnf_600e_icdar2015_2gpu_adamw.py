_base_ = [
    'psenet_mkv2vitae_fpnf_600e_icdar2015_2gpu.py',
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
