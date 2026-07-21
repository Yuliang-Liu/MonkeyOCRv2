#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: install_dflash_vllm.sh --vllm-source DIR [options]

Offline-only installer. It never clones, fetches, pulls, pushes, or uses a
package index. All required source code and wheels must already be local.

Options:
  --vllm-source DIR          Existing local vLLM git worktree (required)
  --backend BACKEND          FLASH_ATTN or FLASHINFER (default: FLASHINFER)
  --flashinfer-wheel-dir DIR Local wheel directory for FlashInfer (optional)
  --patch FILE               Local patch; checked and applied, never generated
  --python PYTHON            Python executable (default: python)
  -h, --help                 Show this help
EOF
}

VLLM_SOURCE=""
BACKEND="FLASHINFER"
FLASHINFER_WHEEL_DIR=""
PATCH_FILE=""
PYTHON_BIN="python"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --vllm-source) VLLM_SOURCE=${2:?missing value}; shift 2 ;;
    --backend) BACKEND=${2:?missing value}; shift 2 ;;
    --flashinfer-wheel-dir) FLASHINFER_WHEEL_DIR=${2:?missing value}; shift 2 ;;
    --patch) PATCH_FILE=${2:?missing value}; shift 2 ;;
    --python) PYTHON_BIN=${2:?missing value}; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -n "$VLLM_SOURCE" ]] || { echo "--vllm-source is required" >&2; exit 2; }
[[ "$BACKEND" == "FLASH_ATTN" || "$BACKEND" == "FLASHINFER" ]] || {
  echo "--backend must be FLASH_ATTN or FLASHINFER" >&2; exit 2;
}
[[ -d "$VLLM_SOURCE/.git" ]] || {
  echo "vLLM source is not an existing git worktree: $VLLM_SOURCE" >&2; exit 1;
}

echo "vLLM source: $VLLM_SOURCE"
echo "vLLM commit: $(git -C "$VLLM_SOURCE" rev-parse HEAD)"
echo "vLLM branch: $(git -C "$VLLM_SOURCE" branch --show-current)"
git -C "$VLLM_SOURCE" status --short
if ! git -C "$VLLM_SOURCE" diff --quiet --ignore-submodules; then
  echo "Refusing to apply/install into a dirty vLLM worktree." >&2
  exit 1
fi

if [[ -n "$PATCH_FILE" ]]; then
  [[ -f "$PATCH_FILE" ]] || { echo "Patch not found: $PATCH_FILE" >&2; exit 1; }
  git -C "$VLLM_SOURCE" apply --check --whitespace=error "$PATCH_FILE"
  git -C "$VLLM_SOURCE" apply --whitespace=error "$PATCH_FILE"
  echo "Applied local patch: $PATCH_FILE"
fi

if [[ "$BACKEND" == "FLASHINFER" ]]; then
  if ! "$PYTHON_BIN" -c 'import flashinfer' >/dev/null 2>&1; then
    [[ -n "$FLASHINFER_WHEEL_DIR" ]] || {
      echo "FlashInfer is unavailable and no local wheel directory was supplied." >&2
      echo "Provide --flashinfer-wheel-dir; no network installation is attempted." >&2
      exit 1
    }
    mapfile -t WHEELS < <(find "$FLASHINFER_WHEEL_DIR" -maxdepth 1 -type f -iname 'flashinfer*.whl' | sort)
    ((${#WHEELS[@]} > 0)) || { echo "No FlashInfer wheel found in $FLASHINFER_WHEEL_DIR" >&2; exit 1; }
    "$PYTHON_BIN" -m pip install --no-index --no-deps "${WHEELS[@]}"
  fi
  "$PYTHON_BIN" -c 'import flashinfer; print("flashinfer import OK")'
fi

"$PYTHON_BIN" -m pip install --no-index --no-deps -e "$VLLM_SOURCE"
"$PYTHON_BIN" -c 'import vllm; print("vLLM:", getattr(vllm, "__version__", "unknown")); print("path:", vllm.__file__)'
"$PYTHON_BIN" -c "from vllm.config import SpeculativeConfig; print('DFlash method available:', 'dflash' in str(SpeculativeConfig))"
echo "Offline installation completed. Run check_dflash_env.py and validate_dflash_backend.py next."
