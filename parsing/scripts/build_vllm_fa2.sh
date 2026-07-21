#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: build_vllm_fa2.sh --vllm-source DIR --flash-attn-source DIR [options]

Build the vLLM FlashAttention 2 extension in the selected local
Python/PyTorch/CUDA environment. This script is offline-only and never copies
an existing .so file.

Options:
  --vllm-source DIR       Existing clean vLLM git worktree (required)
  --flash-attn-source DIR Existing local vllm-flash-attn source (required)
  --python PYTHON         Python executable (default: python)
  --cuda-home DIR         CUDA toolkit root (optional)
  --max-jobs N            Parallel build jobs (default: nproc)
  --jobs N                Alias for --max-jobs
  -h, --help              Show this help
EOF
}

VLLM_SOURCE=""
FLASH_ATTN_SOURCE=""
PYTHON_BIN="python"
CUDA_HOME_ARG=""
MAX_JOBS="$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 1)"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --vllm-source) VLLM_SOURCE=${2:?missing value}; shift 2 ;;
    --flash-attn-source) FLASH_ATTN_SOURCE=${2:?missing value}; shift 2 ;;
    --python) PYTHON_BIN=${2:?missing value}; shift 2 ;;
    --cuda-home) CUDA_HOME_ARG=${2:?missing value}; shift 2 ;;
    --max-jobs|--jobs) MAX_JOBS=${2:?missing value}; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -n "$VLLM_SOURCE" && -d "$VLLM_SOURCE/.git" ]] || {
  echo "--vllm-source must be an existing git worktree" >&2; exit 1;
}
[[ -n "$FLASH_ATTN_SOURCE" && -d "$FLASH_ATTN_SOURCE" ]] || {
  echo "--flash-attn-source must be an existing local source directory" >&2; exit 1;
}
[[ -x "$PYTHON_BIN" || "$(command -v "$PYTHON_BIN" 2>/dev/null || true)" ]] || {
  echo "Python executable not found: $PYTHON_BIN" >&2; exit 1;
}

if [[ -n "$CUDA_HOME_ARG" ]]; then
  export CUDA_HOME="$CUDA_HOME_ARG"
  export PATH="$CUDA_HOME/bin:$PATH"
fi

command -v nvcc >/dev/null || {
  echo "nvcc was not found; provide --cuda-home for the target toolkit" >&2; exit 1;
}
"$PYTHON_BIN" -c 'import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.version.cuda)'
"$PYTHON_BIN" -c 'import ninja; print("ninja", ninja.__file__)'

if ! git -C "$VLLM_SOURCE" diff --quiet --ignore-submodules; then
  echo "Refusing to build from a dirty vLLM worktree." >&2
  git -C "$VLLM_SOURCE" status --short >&2
  exit 1
fi

echo "vLLM commit: $(git -C "$VLLM_SOURCE" rev-parse HEAD)"
echo "vLLM source: $VLLM_SOURCE"
echo "FlashAttention source: $FLASH_ATTN_SOURCE"
echo "CUDA_HOME: ${CUDA_HOME:-auto}"
echo "nvcc: $(nvcc --version | tail -1)"

export VLLM_TARGET_DEVICE=cuda
export VLLM_ENABLE_EXTRA_CUDA_EXTENSIONS=1
export VLLM_FLASH_ATTN_SRC_DIR="$FLASH_ATTN_SOURCE"
export MAX_JOBS
export CMAKE_BUILD_PARALLEL_LEVEL="$MAX_JOBS"

"$PYTHON_BIN" -m pip install \
  --no-index --no-deps --no-build-isolation \
  -e "$VLLM_SOURCE"

"$PYTHON_BIN" - <<'PY'
from vllm.vllm_flash_attn import flash_attn_interface as fa2

if not fa2.FA2_AVAILABLE:
    raise SystemExit(f"FA2 extension import failed: {fa2.FA2_UNAVAILABLE_REASON}")
print("FA2 extension import: PASS")
print("FA2 module:", fa2._vllm_fa2_C.__file__)
print("Runtime image validation is still required; run validate_dflash_backend.py.")
PY
