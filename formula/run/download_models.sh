#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "${ROOT_DIR}"

HF_CMD=${HF_CMD:-hf}
UNIMERNET_TINY_REPO=${UNIMERNET_TINY_REPO:-wanderkid/unimernet_tiny}
MONKEY_ENCODER_REPO=${MONKEY_ENCODER_REPO:-zenosai/MonkeyOCRv2-S}
FORMULA_CKPT_REPO=${FORMULA_CKPT_REPO:-zhd36/monkeyocrv2_s_formula}
FORMULA_CKPT_FILE=${FORMULA_CKPT_FILE:-monkeyocrv2_s_formula.pth}

mkdir -p models model_weight

"${HF_CMD}" download "${UNIMERNET_TINY_REPO}" --local-dir models/unimernet_tiny
"${HF_CMD}" download "${MONKEY_ENCODER_REPO}" --local-dir models/monkeyocrv2_vit
"${HF_CMD}" download "${FORMULA_CKPT_REPO}" --include "${FORMULA_CKPT_FILE}" --local-dir model_weight
