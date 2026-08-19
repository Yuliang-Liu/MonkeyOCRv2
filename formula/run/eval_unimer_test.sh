#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "${ROOT_DIR}"

PYTHON_CMD=${PYTHON_CMD:-python}
read -r -a PYTHON_ARR <<< "${PYTHON_CMD}"

MODEL_NAME=${MODEL_NAME:-monkeyocrv2_s}
CONFIG=${CONFIG:-configs/eval/monkeyocrv2_s.yaml}
CHECKPOINT=${CHECKPOINT:-model_weight/monkeyocrv2_s_formula.pth}
DATA_ROOT=${DATA_ROOT:-data/UniMER-Test}
DATASETS=${DATASETS:-"cpe hwe sce spe"}
BATCH_SIZE=${BATCH_SIZE:-256}
NUM_WORKERS=${NUM_WORKERS:-8}
DEVICE=${DEVICE:-cuda}
POOLS=${POOLS:-64}

mkdir -p eval_results/unimer_test/cdm_inputs eval_results/unimer_test/cdm_outputs eval_results/unimer_test/reports eval_results/unimer_test/logs

"${PYTHON_ARR[@]}" scripts/eval_unimer_test.py \
  --cfg-path "${CONFIG}" \
  --exp-name "${MODEL_NAME}" \
  --output-dir eval_results/unimer_test/cdm_inputs \
  --data-root "${DATA_ROOT}" \
  --datasets ${DATASETS} \
  --batch-size "${BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}" \
  --device "${DEVICE}" \
  --options model.finetuned="${CHECKPOINT}" "$@"

if [[ "${RUN_CDM:-1}" == "1" ]]; then
  export MPLCONFIGDIR="${MPLCONFIGDIR:-${ROOT_DIR}/eval_results/unimer_test/tmp/matplotlib}"
  export CDM_SAVE_MATCH_VIS="${CDM_SAVE_MATCH_VIS:-0}"
  export CDM_SAVE_TOKEN_VIS="${CDM_SAVE_TOKEN_VIS:-0}"
  export CDM_PDFLATEX_TIMEOUT="${CDM_PDFLATEX_TIMEOUT:-60}"
  mkdir -p "${MPLCONFIGDIR}"
  for input in eval_results/unimer_test/cdm_inputs/${MODEL_NAME}__*.json; do
    name=$(basename "${input}" .json)
    "${PYTHON_ARR[@]}" cdm_local/evaluation.py \
      -i "${input}" \
      -o eval_results/unimer_test/cdm_outputs \
      -p "${POOLS}" 2>&1 | tee "eval_results/unimer_test/logs/cdm_${name}.log"
  done
  "${PYTHON_ARR[@]}" scripts/summarize_cdm.py \
    --root eval_results/unimer_test/cdm_outputs \
    --input-root eval_results/unimer_test/cdm_inputs \
    --output eval_results/unimer_test/reports/cdm_summary.json \
    --markdown-output eval_results/unimer_test/reports/cdm_summary.md \
    --title "UniMER-Test CDM Summary"
fi
