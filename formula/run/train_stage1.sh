#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "${ROOT_DIR}"

GPUS=${GPUS:-0,1,2,3}
NPROC=${NPROC:-4}
CONFIG=${CONFIG:-configs/train/monkeyocrv2_s_stage1_freeze50k.yaml}
LOG_DIR=${LOG_DIR:-logs_train}
mkdir -p "${LOG_DIR}"

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${GPUS}"
export TOKENIZERS_PARALLELISM=false

OPT_ARGS=()
if [[ "$#" -gt 0 ]]; then
  OPT_ARGS=(--options "$@")
fi

torchrun --nproc_per_node="${NPROC}" train.py \
  --cfg-path "${CONFIG}" \
  "${OPT_ARGS[@]}" \
  2>&1 | tee -a "${LOG_DIR}/monkeyocrv2_s_stage1_freeze50k.log"
