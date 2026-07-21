#!/usr/bin/env bash
set -euo pipefail

# Generic MonkeyOCRv2 DFlash + FlashAttention launcher.
# The FA2 extension must be built in the same Python/PyTorch/CUDA environment;
# this script never copies a compiled extension.
if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Usage: run_dflash_flashattn.sh [additional vLLM arguments]

Required environment variables:
  TARGET_MODEL       Local MonkeyOCRv2 target checkpoint
  DRAFT_MODEL        Local MonkeyOCRv2 DFlash draft checkpoint

Optional environment variables:
  VLLM_SOURCE, PYTHON_BIN, CUDA_VISIBLE_DEVICES, PORT, MAX_NUM_SEQS,
  MAX_NUM_BATCHED_TOKENS, MAX_MODEL_LEN, NUM_SPECULATIVE_TOKENS,
  GPU_MEMORY_UTILIZATION, TENSOR_PARALLEL_SIZE
EOF
  exit 0
fi
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
TARGET_MODEL=${TARGET_MODEL:-}
DRAFT_MODEL=${DRAFT_MODEL:-}
VLLM_SOURCE=${VLLM_SOURCE:-}
PYTHON_BIN=${PYTHON_BIN:-python}
PORT=${PORT:-8888}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-128}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-65536}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-16384}
NUM_SPECULATIVE_TOKENS=${NUM_SPECULATIVE_TOKENS:-16}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.5}
TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE:-1}

[[ -n "$TARGET_MODEL" ]] || { echo "TARGET_MODEL is required" >&2; exit 2; }
[[ -n "$DRAFT_MODEL" ]] || { echo "DRAFT_MODEL is required" >&2; exit 2; }
[[ -d "$TARGET_MODEL" ]] || { echo "target model not found: $TARGET_MODEL" >&2; exit 1; }
[[ -d "$DRAFT_MODEL" ]] || { echo "draft model not found: $DRAFT_MODEL" >&2; exit 1; }

if [[ -n "$VLLM_SOURCE" ]]; then
  export PYTHONPATH="$VLLM_SOURCE:${PYTHONPATH:-}"
fi
export PYTHONPATH="$ROOT/parsing:${PYTHONPATH:-}"

exec "$PYTHON_BIN" "$ROOT/parsing/serve.py" \
  --model-path "$TARGET_MODEL" \
  --draft-model "$DRAFT_MODEL" \
  --target-attention-backend FLASH_ATTN \
  --dflash-attention-backend FLASH_ATTN \
  --num-speculative-tokens "$NUM_SPECULATIVE_TOKENS" \
  --dflash-max-num-seqs "$MAX_NUM_SEQS" \
  --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
  --served-model-name "${SERVED_MODEL_NAME:-MonkeyOCRv2}" \
  --port "$PORT" \
  "$@"
