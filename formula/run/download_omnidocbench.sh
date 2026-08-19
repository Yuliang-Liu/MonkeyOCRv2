#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "${ROOT_DIR}"

HF_CMD=${HF_CMD:-hf}
ODB_REPO=${ODB_REPO:-opendatalab/OmniDocBench}
OUT_DIR=${OUT_DIR:-data/omnidocbench}

mkdir -p "${OUT_DIR}"
"${HF_CMD}" download "${ODB_REPO}" --repo-type dataset --local-dir "${OUT_DIR}"

