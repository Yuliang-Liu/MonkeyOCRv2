#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "${ROOT_DIR}"

GPUS=${GPUS:-0,1,2,3}
NPROC=${NPROC:-4}
CONFIG=${CONFIG:-configs/train/monkeyocrv2_s_stage2_unfreeze250k.yaml}
LOG_DIR=${LOG_DIR:-logs_train}
mkdir -p "${LOG_DIR}"

STAGE1_CKPT=${STAGE1_CKPT:-}
if [[ -z "${STAGE1_CKPT}" ]]; then
  STAGE1_CKPT=$(find outputs/monkeyocrv2_s_stage1_freeze50k -path '*/checkpoint_latest.pth' -type f 2>/dev/null | sort | tail -n 1 || true)
fi
if [[ -z "${STAGE1_CKPT}" ]]; then
  echo "Missing stage-1 checkpoint. Pass STAGE1_CKPT=/path/to/checkpoint_latest.pth." >&2
  exit 2
fi

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${GPUS}"
export TOKENIZERS_PARALLELISM=false

OPT_ARGS=(--options model.finetuned="${STAGE1_CKPT}")
if [[ "$#" -gt 0 ]]; then
  OPT_ARGS+=("$@")
fi

torchrun --nproc_per_node="${NPROC}" train.py \
  --cfg-path "${CONFIG}" \
  "${OPT_ARGS[@]}" \
  2>&1 | tee -a "${LOG_DIR}/monkeyocrv2_s_stage2_unfreeze250k.log"
