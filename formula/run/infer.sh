#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "${ROOT_DIR}"

PYTHON_CMD=${PYTHON_CMD:-python}
read -r -a PYTHON_ARR <<< "${PYTHON_CMD}"

"${PYTHON_ARR[@]}" scripts/infer.py "$@"

