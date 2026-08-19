#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "${ROOT_DIR}"

bash run/eval_unimer_test.sh
bash run/eval_mathwriting.sh
bash run/eval_omnidocbench.sh
