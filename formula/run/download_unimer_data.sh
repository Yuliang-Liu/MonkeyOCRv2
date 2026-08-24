#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "${ROOT_DIR}"

HF_CMD=${HF_CMD:-hf}
UNIMER_REPO=${UNIMER_REPO:-wanderkid/UniMER_Dataset}
OUT_DIR=${OUT_DIR:-data}

mkdir -p "${OUT_DIR}"
"${HF_CMD}" download "${UNIMER_REPO}" --repo-type dataset --local-dir "${OUT_DIR}"

cat <<'EOF'
UniMER files have been downloaded.

If needed, arrange or extract them to the following layout before training/evaluation:
  data/UniMER1M/images/*.png
  data/UniMER1M/train.txt
  data/UniMER-Test/cpe/*.png
  data/UniMER-Test/cpe.txt
  data/UniMER-Test/hwe/*.png
  data/UniMER-Test/hwe.txt
  data/UniMER-Test/sce/*.png
  data/UniMER-Test/sce.txt
  data/UniMER-Test/spe/*.png
  data/UniMER-Test/spe.txt

The released training recipe also uses HME100K. Prepare it with:
  HME100K_ARCHIVE=/path/to/HME100K.zip bash run/prepare_train_data.sh
EOF
