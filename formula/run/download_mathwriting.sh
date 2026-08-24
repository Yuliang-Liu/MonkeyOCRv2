#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "${ROOT_DIR}"

ARCHIVE_URL=${ARCHIVE_URL:-https://storage.googleapis.com/mathwriting_data/mathwriting-2024.tgz}
ARCHIVE_PATH=${ARCHIVE_PATH:-data/mathwriting-2024.tgz}
EXTRACT_DIR=${EXTRACT_DIR:-data}

mkdir -p "$(dirname "${ARCHIVE_PATH}")" "${EXTRACT_DIR}"
if [[ ! -f "${ARCHIVE_PATH}" ]]; then
  curl -L -C - --fail --retry 3 "${ARCHIVE_URL}" -o "${ARCHIVE_PATH}"
fi
tar -xzf "${ARCHIVE_PATH}" -C "${EXTRACT_DIR}"

