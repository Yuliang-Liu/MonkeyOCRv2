#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "${ROOT_DIR}"

PYTHON_CMD=${PYTHON_CMD:-python}
read -r -a PYTHON_ARR <<< "${PYTHON_CMD}"

HME100K_ARCHIVE=${HME100K_ARCHIVE:-data/archive.zip}
UNIMER1M_ROOT=${UNIMER1M_ROOT:-data/UniMER1M}
MERGED_ROOT=${MERGED_ROOT:-data/UniMER1M_HME100K_merged}

"${PYTHON_ARR[@]}" scripts/prepare_hme100k_issue14.py \
  --repo-root "${ROOT_DIR}" \
  --archive "${HME100K_ARCHIVE}" \
  --unimer-root "${UNIMER1M_ROOT}" \
  --merged-root "${MERGED_ROOT}" \
  --reset-merged-output \
  "$@"
