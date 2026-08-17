#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
ROOT_DIR=.
OUT_DIR="${1:-$ROOT_DIR/outputs/crnn_chinese_stage1/train_$(date '+%F_%H-%M-%S')}"

mkdir -p "$OUT_DIR"
stdbuf -oL -eL python "$ROOT_DIR/train.py" \
  model=crnn_monkey \
  dataset=chinese_monkey \
  charset=ppocr_keys_v1 \
  seed=3407 \
  trainer.devices=1 \
  model.batch_size=1024 \
  data.num_workers=8 \
  trainer.max_epochs=30 \
  +trainer.num_sanity_val_steps=0 \
  +trainer.check_val_every_n_epoch=5 \
  +trainer.log_every_n_steps=10 \
  model.freeze_mode=freeze_encoder \
  hydra.run.dir="$OUT_DIR" 2>&1 | tee "$OUT_DIR/train.log"
