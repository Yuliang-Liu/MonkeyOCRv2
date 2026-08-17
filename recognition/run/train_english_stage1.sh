#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
ROOT_DIR=.
OUT_DIR="${1:-$ROOT_DIR/outputs/english_stage1/train_$(date '+%F_%H-%M-%S')}"
BATCH_SIZE="${2:-1024}"
VAL_CHECK_INTERVAL="${3:-3000}"
mkdir -p "$OUT_DIR"
stdbuf -oL -eL python "$ROOT_DIR/train.py"   model=parseq_monkey_multiscale   dataset=english_monkey   charset=94_full   seed=3407   trainer.devices=1   model.batch_size="$BATCH_SIZE"   data.num_workers=8   trainer.max_epochs=20   trainer.val_check_interval="$VAL_CHECK_INTERVAL"   model.freeze_mode=freeze_encoder   hydra.run.dir="$OUT_DIR" 2>&1 | tee "$OUT_DIR/train.log"
