#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
DEFAULT_PATCH="$ROOT/parsing/patches/vllm-dflash.patch"
DEFAULT_BASE_COMMIT="dbc3d9991ab0e5adc0db6a8c71c9059268032a14"

usage() {
  cat <<'EOF'
Usage: install_dflash_vllm.sh --vllm-source DIR [options]

Offline-only installer. Source code and wheels must already be available locally.

Options:
  --vllm-source DIR          Clean vLLM git worktree (required)
  --patch FILE               DFlash patch (default: parsing/patches/vllm-dflash.patch)
  --base-commit SHA          Expected clean vLLM commit
  --python PYTHON            Python executable (default: python)
  --skip-fa2-build           Do not build FA2 (only for a prevalidated environment)
  --flash-attn-source DIR    Local vllm-flash-attn source for FA2 build
  --cuda-home DIR            CUDA toolkit for FA2 build
  --jobs N                   FA2 build parallelism
  -h, --help                 Show this help
EOF
}

VLLM_SOURCE=""
PATCH_FILE="$DEFAULT_PATCH"
BASE_COMMIT="$DEFAULT_BASE_COMMIT"
PYTHON_BIN="python"
SKIP_FA2_BUILD=0
FLASH_ATTN_SOURCE=""
CUDA_HOME_ARG=""
JOBS=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --vllm-source) VLLM_SOURCE=${2:?missing value}; shift 2 ;;
    --patch) PATCH_FILE=${2:?missing value}; shift 2 ;;
    --base-commit) BASE_COMMIT=${2:?missing value}; shift 2 ;;
    --python) PYTHON_BIN=${2:?missing value}; shift 2 ;;
    --skip-fa2-build) SKIP_FA2_BUILD=1; shift ;;
    --flash-attn-source) FLASH_ATTN_SOURCE=${2:?missing value}; shift 2 ;;
    --cuda-home) CUDA_HOME_ARG=${2:?missing value}; shift 2 ;;
    --jobs) JOBS=${2:?missing value}; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -n "$VLLM_SOURCE" ]] || { echo "--vllm-source is required" >&2; exit 2; }
[[ -d "$VLLM_SOURCE/.git" ]] || {
  echo "vLLM source is not a git worktree: $VLLM_SOURCE" >&2
  exit 1
}
[[ -f "$PATCH_FILE" ]] || { echo "Patch not found: $PATCH_FILE" >&2; exit 1; }

ACTUAL_COMMIT=$(git -C "$VLLM_SOURCE" rev-parse HEAD)
[[ "$ACTUAL_COMMIT" == "$BASE_COMMIT" ]] || {
  echo "vLLM base commit mismatch: expected $BASE_COMMIT, got $ACTUAL_COMMIT" >&2
  exit 1
}
if ! git -C "$VLLM_SOURCE" diff --quiet --ignore-submodules; then
  echo "Refusing to apply patch to a dirty vLLM worktree." >&2
  exit 1
fi

echo "vLLM source: $VLLM_SOURCE"
echo "vLLM base commit: $ACTUAL_COMMIT"
git -C "$VLLM_SOURCE" apply --check --whitespace=error "$PATCH_FILE"
git -C "$VLLM_SOURCE" apply --whitespace=error "$PATCH_FILE"
echo "DFlash patch applied."

if [[ "$SKIP_FA2_BUILD" -eq 0 ]]; then
  [[ -n "$FLASH_ATTN_SOURCE" ]] || { echo "--flash-attn-source is required for FA2 build." >&2; exit 1; }
  [[ -n "$CUDA_HOME_ARG" ]] || { echo "--cuda-home is required for FA2 build." >&2; exit 1; }
  BUILD_ARGS=(
    --vllm-source "$VLLM_SOURCE"
    --flash-attn-source "$FLASH_ATTN_SOURCE"
    --python "$PYTHON_BIN"
    --cuda-home "$CUDA_HOME_ARG"
    --allow-patched-worktree
  )
  [[ -n "$JOBS" ]] && BUILD_ARGS+=(--jobs "$JOBS")
  bash "$ROOT/parsing/scripts/build_vllm_fa2.sh" "${BUILD_ARGS[@]}"
fi

"$PYTHON_BIN" -m pip install --no-index --no-deps --no-build-isolation -e "$VLLM_SOURCE"
"$PYTHON_BIN" -c 'import vllm; print("vLLM:", getattr(vllm, "__version__", "unknown")); print("path:", vllm.__file__)'
"$PYTHON_BIN" -c 'from vllm.config import SpeculativeConfig; print("DFlash config import OK:", hasattr(SpeculativeConfig, "is_dflash"))'
echo "Offline DFlash vLLM installation completed."
