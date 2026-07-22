#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PYTHON_BIN="python"
VLLM_VERSION="0.25.1"
INDEX_URL=""
TARGET_MODEL=""
DRAFT_MODEL=""

usage() {
  cat <<'EOF'
Usage: install_dflash_vllm_pip.sh [options]

Install and validate a native pip vLLM DFlash environment. This path does
not apply the bundled source patch and does not compile FlashAttention.

Options:
  --vllm-version VERSION  vLLM version with native DFlash (default: 0.25.1)
  --python PYTHON         Python executable (default: python)
  --index-url URL         Optional pip index URL
  --target-model DIR      Optional MonkeyOCRv2 target model to validate
  --draft-model DIR       Optional DFlash draft model to validate
  -h, --help              Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --vllm-version) VLLM_VERSION=${2:?missing value}; shift 2 ;;
    --python) PYTHON_BIN=${2:?missing value}; shift 2 ;;
    --index-url) INDEX_URL=${2:?missing value}; shift 2 ;;
    --target-model) TARGET_MODEL=${2:?missing value}; shift 2 ;;
    --draft-model) DRAFT_MODEL=${2:?missing value}; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

INSTALLED_VERSION=$("${PYTHON_BIN}" - <<'PY' 2>/dev/null || true
try:
    import vllm
    print(vllm.__version__)
except Exception:
    pass
PY
)

if [[ "$INSTALLED_VERSION" == "$VLLM_VERSION" ]]; then
  echo "vLLM ${VLLM_VERSION} is already installed; skipping pip install."
else
  PIP_ARGS=("${PYTHON_BIN}" -m pip install "vllm==${VLLM_VERSION}")
  [[ -n "$INDEX_URL" ]] && PIP_ARGS+=(--index-url "$INDEX_URL")
  "${PIP_ARGS[@]}"
fi

CHECK_ARGS=("${PYTHON_BIN}" "$ROOT/parsing/scripts/check_dflash_env.py" "--require-native-dflash")
[[ -n "$TARGET_MODEL" ]] && CHECK_ARGS+=(--target-model "$TARGET_MODEL")
[[ -n "$DRAFT_MODEL" ]] && CHECK_ARGS+=(--draft-model "$DRAFT_MODEL")
"${CHECK_ARGS[@]}"

echo "Native pip vLLM DFlash installation and validation completed."
