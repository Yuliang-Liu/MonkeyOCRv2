#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
ROOT_DIR=.
STAGE1_OUT="${1:?stage1 output dir required}"
OUT_DIR="${2:-$ROOT_DIR/outputs/chinese_stage2/train_$(date '+%F_%H-%M-%S')}"
LR="${3:-0.0007}"
mkdir -p "$OUT_DIR"
stdbuf -oL -eL python "$ROOT_DIR/train.py"   model=parseq_monkey_multiscale   dataset=chinese_monkey   charset=ppocr_keys_v1   pretrained="$STAGE1_OUT/checkpoints/last.ckpt"   seed=3407   trainer.devices=1   model.batch_size=512   model.lr="$LR"   data.num_workers=8   trainer.max_epochs=70   trainer.val_check_interval=null   +trainer.num_sanity_val_steps=0   +trainer.check_val_every_n_epoch=10   +trainer.log_every_n_steps=1   model.freeze_mode=none   hydra.run.dir="$OUT_DIR" 2>&1 | tee "$OUT_DIR/train.log"
