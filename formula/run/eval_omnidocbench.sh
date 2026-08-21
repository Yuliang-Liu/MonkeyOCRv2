#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "${ROOT_DIR}"

PYTHON_CMD=${PYTHON_CMD:-python}
read -r -a PYTHON_ARR <<< "${PYTHON_CMD}"

MODEL_NAME=${MODEL_NAME:-monkeyocrv2_s}
CONFIG=${CONFIG:-configs/eval/monkeyocrv2_s.yaml}
CHECKPOINT=${CHECKPOINT:-model_weight/monkeyocrv2_s_formula.pth}
ODB_VERSION=${ODB_VERSION:-opendatalab_v1_6}
ODB_JSON=${ODB_JSON:-data/omnidocbench/OmniDocBench.json}
ODB_IMAGE_ROOT=${ODB_IMAGE_ROOT:-data/omnidocbench/images}
BATCH_SIZE=${BATCH_SIZE:-64}
NUM_WORKERS=${NUM_WORKERS:-8}
DEVICE=${DEVICE:-cuda}
POOLS=${POOLS:-32}
LIMIT=${LIMIT:-0}

LIMIT_ARG=()
if [[ "${LIMIT}" != "0" ]]; then
  LIMIT_ARG=(--limit "${LIMIT}")
fi

MANIFEST="data/omnidocbench/manifests/${ODB_VERSION}_formula.jsonl"
CROP_DIR="data/omnidocbench/crops/${ODB_VERSION}_formula"
PRED_DIR="eval_results/omnidocbench/predictions/${MODEL_NAME}"
REPORT_DIR="eval_results/omnidocbench/reports/${MODEL_NAME}"
CDM_INPUT_DIR="eval_results/omnidocbench/cdm_inputs"
CDM_OUTPUT_DIR="eval_results/omnidocbench/cdm_outputs"

mkdir -p "${PRED_DIR}" "${REPORT_DIR}" "${CDM_INPUT_DIR}" "${CDM_OUTPUT_DIR}" eval_results/omnidocbench/logs

if [[ "${PREPARE_DATA:-1}" == "1" ]]; then
  "${PYTHON_ARR[@]}" scripts/omnidocbench/prepare_formula_crops.py \
    --annotation "${ODB_JSON}" \
    --image-root "${ODB_IMAGE_ROOT}" \
    --out-dir "${CROP_DIR}" \
    --manifest "${MANIFEST}" \
    --category equation_isolated \
    --gt-key latex \
    "${LIMIT_ARG[@]}"
fi

PREDICTIONS="${PRED_DIR}/${ODB_VERSION}_predictions.jsonl"
"${PYTHON_ARR[@]}" scripts/omnidocbench/predict_unimernet.py \
  --repo-root "${ROOT_DIR}" \
  --cfg-path "${CONFIG}" \
  --manifest "${MANIFEST}" \
  --output "${PREDICTIONS}" \
  --batch-size "${BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}" \
  --device "${DEVICE}" \
  --cfg-options model.finetuned="${CHECKPOINT}" \
  "${LIMIT_ARG[@]}"

"${PYTHON_ARR[@]}" scripts/omnidocbench/score_text_local.py \
  --predictions "${PREDICTIONS}" \
  --output "${REPORT_DIR}/${ODB_VERSION}_local_text_score.json" \
  --repo-root "${ROOT_DIR}"

MERGED_JSON="${PRED_DIR}/${MODEL_NAME}_${ODB_VERSION}_formula_pred.json"
"${PYTHON_ARR[@]}" scripts/omnidocbench/merge_predictions.py \
  --annotation "${ODB_JSON}" \
  --manifest "${MANIFEST}" \
  --predictions "${PREDICTIONS}" \
  --output "${MERGED_JSON}" \
  --prediction-field pred

OFFICIAL_CONFIG="${PRED_DIR}/${MODEL_NAME}_${ODB_VERSION}_formula_recognition.yaml"
sed "s#__PREDICTED_JSON__#${MERGED_JSON}#g" \
  configs/eval/omnidocbench_formula_recognition.template.yaml > "${OFFICIAL_CONFIG}"

OFFICIAL_CDM_INPUT="${REPORT_DIR}/${ODB_VERSION}_cdm_input_formula.json"
if [[ "${RUN_OFFICIAL:-0}" == "1" ]]; then
  if [[ -z "${ODB_EVAL_REPO:-}" || ! -f "${ODB_EVAL_REPO}/pdf_validation.py" ]]; then
    echo "RUN_OFFICIAL=1 requires ODB_EVAL_REPO=/path/to/OmniDocBench with pdf_validation.py." >&2
    exit 2
  fi
  pushd "${ODB_EVAL_REPO}" >/dev/null
  PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}" "${PYTHON_ARR[@]}" pdf_validation.py --config "${ROOT_DIR}/${OFFICIAL_CONFIG}"
  popd >/dev/null

  SAVE_STEM=$(basename "${MERGED_JSON%.*}")
  cp "${ODB_EVAL_REPO}/result/${SAVE_STEM}_metric_result.json" "${REPORT_DIR}/${ODB_VERSION}_official_metric_result.json"
  cp "${ODB_EVAL_REPO}/result/${SAVE_STEM}_per_page_edit.json" "${REPORT_DIR}/${ODB_VERSION}_official_per_page_edit.json"
  cp "${ODB_EVAL_REPO}/result/${SAVE_STEM}_formula.json" "${OFFICIAL_CDM_INPUT}"
fi

if [[ "${RUN_CDM:-1}" == "1" ]]; then
  CDM_INPUT="${CDM_INPUT_DIR}/${MODEL_NAME}__${ODB_VERSION}.json"
  if [[ -f "${OFFICIAL_CDM_INPUT}" ]]; then
    cp "${OFFICIAL_CDM_INPUT}" "${CDM_INPUT}"
  else
    "${PYTHON_ARR[@]}" scripts/mathwriting/build_cdm_input.py \
      --predictions "${PREDICTIONS}" \
      --output "${CDM_INPUT}" \
      --model-name "${MODEL_NAME}" \
      --normalization "${NORMALIZATION:-cdm}"
  fi

  export MPLCONFIGDIR="${MPLCONFIGDIR:-${ROOT_DIR}/eval_results/omnidocbench/tmp/matplotlib}"
  export CDM_SAVE_MATCH_VIS="${CDM_SAVE_MATCH_VIS:-0}"
  export CDM_SAVE_TOKEN_VIS="${CDM_SAVE_TOKEN_VIS:-0}"
  export CDM_PDFLATEX_TIMEOUT="${CDM_PDFLATEX_TIMEOUT:-60}"
  mkdir -p "${MPLCONFIGDIR}"
  "${PYTHON_ARR[@]}" cdm_local/evaluation.py \
    -i "${CDM_INPUT}" \
    -o "${CDM_OUTPUT_DIR}" \
    -p "${POOLS}" 2>&1 | tee "eval_results/omnidocbench/logs/cdm_${MODEL_NAME}__${ODB_VERSION}.log"

  "${PYTHON_ARR[@]}" scripts/summarize_cdm.py \
    --root "${CDM_OUTPUT_DIR}" \
    --input-root "${CDM_INPUT_DIR}" \
    --output eval_results/omnidocbench/reports/cdm_summary.json \
    --markdown-output eval_results/omnidocbench/reports/cdm_summary.md \
    --title "OmniDocBench CDM Summary"
fi

