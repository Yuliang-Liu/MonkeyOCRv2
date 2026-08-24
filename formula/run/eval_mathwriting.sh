#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "${ROOT_DIR}"

PYTHON_CMD=${PYTHON_CMD:-python}
read -r -a PYTHON_ARR <<< "${PYTHON_CMD}"

MODEL_NAME=${MODEL_NAME:-monkeyocrv2_s}
CONFIG=${CONFIG:-configs/eval/monkeyocrv2_s.yaml}
CHECKPOINT=${CHECKPOINT:-model_weight/monkeyocrv2_s_formula.pth}
DATA_ROOT=${DATA_ROOT:-data/mathwriting-2024}
MANIFEST=${MANIFEST:-data/mathwriting/manifests/mathwriting_test.jsonl}
IMAGE_DIR=${IMAGE_DIR:-data/mathwriting/rendered}
BATCH_SIZE=${BATCH_SIZE:-128}
NUM_WORKERS=${NUM_WORKERS:-8}
DEVICE=${DEVICE:-cuda}
POOLS=${POOLS:-64}
LIMIT=${LIMIT:-0}

LIMIT_ARG=()
if [[ "${LIMIT}" != "0" ]]; then
  LIMIT_ARG=(--limit "${LIMIT}")
fi

mkdir -p eval_results/mathwriting/predictions/${MODEL_NAME} eval_results/mathwriting/reports/${MODEL_NAME} eval_results/mathwriting/cdm_inputs eval_results/mathwriting/cdm_outputs eval_results/mathwriting/logs

if [[ "${PREPARE_DATA:-1}" == "1" ]]; then
  "${PYTHON_ARR[@]}" scripts/mathwriting/prepare_mathwriting.py \
    --data-root "${DATA_ROOT}" \
    --splits test \
    --manifest "${MANIFEST}" \
    --image-dir "${IMAGE_DIR}" \
    --num-workers "${NUM_WORKERS}" \
    "${LIMIT_ARG[@]}"
fi

PREDICTIONS="eval_results/mathwriting/predictions/${MODEL_NAME}/mathwriting_test_predictions.jsonl"
"${PYTHON_ARR[@]}" scripts/mathwriting/predict_unimernet.py \
  --repo-root "${ROOT_DIR}" \
  --cfg-path "${CONFIG}" \
  --manifest "${MANIFEST}" \
  --output "${PREDICTIONS}" \
  --batch-size "${BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}" \
  --device "${DEVICE}" \
  --cfg-options model.finetuned="${CHECKPOINT}" \
  "${LIMIT_ARG[@]}"

"${PYTHON_ARR[@]}" scripts/mathwriting/score_mathwriting.py \
  --repo-root "${ROOT_DIR}" \
  --manifest "${MANIFEST}" \
  --predictions "${PREDICTIONS}" \
  --output "eval_results/mathwriting/reports/${MODEL_NAME}/mathwriting_test_score.json" \
  --per-sample-output "eval_results/mathwriting/reports/${MODEL_NAME}/mathwriting_test_per_sample.jsonl"

if [[ "${RUN_CDM:-1}" == "1" ]]; then
  CDM_INPUT="eval_results/mathwriting/cdm_inputs/${MODEL_NAME}__mathwriting_test.json"
  "${PYTHON_ARR[@]}" scripts/mathwriting/build_cdm_input.py \
    --predictions "${PREDICTIONS}" \
    --output "${CDM_INPUT}" \
    --model-name "${MODEL_NAME}" \
    --normalization "${NORMALIZATION:-cdm}"

  export MPLCONFIGDIR="${MPLCONFIGDIR:-${ROOT_DIR}/eval_results/mathwriting/tmp/matplotlib}"
  export CDM_SAVE_MATCH_VIS="${CDM_SAVE_MATCH_VIS:-0}"
  export CDM_SAVE_TOKEN_VIS="${CDM_SAVE_TOKEN_VIS:-0}"
  export CDM_PDFLATEX_TIMEOUT="${CDM_PDFLATEX_TIMEOUT:-60}"
  mkdir -p "${MPLCONFIGDIR}"
  "${PYTHON_ARR[@]}" cdm_local/evaluation.py \
    -i "${CDM_INPUT}" \
    -o eval_results/mathwriting/cdm_outputs \
    -p "${POOLS}" 2>&1 | tee "eval_results/mathwriting/logs/cdm_${MODEL_NAME}__mathwriting_test.log"

  "${PYTHON_ARR[@]}" scripts/summarize_cdm.py \
    --root eval_results/mathwriting/cdm_outputs \
    --input-root eval_results/mathwriting/cdm_inputs \
    --output eval_results/mathwriting/reports/cdm_summary.json \
    --markdown-output eval_results/mathwriting/reports/cdm_summary.md \
    --title "MathWriting CDM Summary"
fi

