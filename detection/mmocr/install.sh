#!/bin/bash
# Install the MonkeyOCRv2 text-detection add-on on top of the official
# MMOCR v1.0.1 release.
#
# Usage:
#   bash install.sh [TARGET_MMOCR_DIR]      # default: ./mmocr
#
# The script clones the official MMOCR repository at tag v1.0.1 (unless the
# target directory already contains a checkout) and copies the add-on files
# into it, then installs MMOCR in editable mode.
#
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MMOCR_DIR="${1:-$SCRIPT_DIR/mmocr}"
MMOCR_GIT="${MMOCR_GIT:-https://github.com/open-mmlab/mmocr.git}"
MMOCR_TAG="${MMOCR_TAG:-v1.0.1}"

if [ ! -d "$MMOCR_DIR/.git" ]; then
    echo "Cloning MMOCR ($MMOCR_TAG) from $MMOCR_GIT ..."
    git clone "$MMOCR_GIT" "$MMOCR_DIR"
    git -C "$MMOCR_DIR" checkout "$MMOCR_TAG"
else
    echo "Using existing MMOCR checkout at $MMOCR_DIR"
fi

echo "Copying add-on files ..."
cp -r "$SCRIPT_DIR/configs/." "$MMOCR_DIR/configs/"
cp -r "$SCRIPT_DIR/patch/." "$MMOCR_DIR/mmocr/"
cp -r "$SCRIPT_DIR/dataset_zoo/." "$MMOCR_DIR/dataset_zoo/"
cp -r "$SCRIPT_DIR/tools/." "$MMOCR_DIR/tools/"

echo "Installing MMOCR (editable) and extra dependencies ..."
# mmocr/__init__.py imports mmdet unconditionally, but mmdet is not listed in
# MMOCR's runtime requirements. 3.1.0 is the newest release compatible with
# MMCV 2.0.x.
pip install mmdet==3.1.0
pip install -e "$MMOCR_DIR"
pip install -r "$MMOCR_DIR/requirements/runtime.txt"
# Required by the MonkeyOCRv2ViTAEBackbone
pip install "transformers==4.57.1" "safetensors==0.7.0"
# Required by TensorboardVisBackend in the training configs
pip install future tensorboard

# imgaug 0.4.0 (a runtime dependency of MMOCR) is incompatible with NumPy >= 2:
# it accesses np.sctypes, which was removed in NumPy 2.0. Patch it in place.
python - << 'IMGFIX'
import importlib.util
import os

import numpy as np
if hasattr(np, 'sctypes'):
    print('NumPy 1.x detected, no imgaug patch needed')
else:
    # Locate imgaug/imgaug.py without importing it - on NumPy 2.x the import
    # is exactly what fails, so `import imgaug` here would abort the patch.
    spec = importlib.util.find_spec('imgaug')
    src = os.path.join(os.path.dirname(spec.origin), 'imgaug.py')
    text = open(src).read()
    old = ('NP_FLOAT_TYPES = set(np.sctypes["float"])\n'
           'NP_INT_TYPES = set(np.sctypes["int"])\n'
           'NP_UINT_TYPES = set(np.sctypes["uint"])')
    new = ('_np_sctypes = {\n'
           '    "float": (np.float16, np.float32, np.float64, np.longdouble),\n'
           '    "int": (np.int8, np.int16, np.int32, np.int64),\n'
           '    "uint": (np.uint8, np.uint16, np.uint32, np.uint64),\n'
           '}\n'
           'NP_FLOAT_TYPES = set(_np_sctypes["float"])\n'
           'NP_INT_TYPES = set(_np_sctypes["int"])\n'
           'NP_UINT_TYPES = set(_np_sctypes["uint"])')
    if old in text:
        open(src, 'w').write(text.replace(old, new))
        print('imgaug patched for NumPy 2.x')
    else:
        print('imgaug already patched or unexpected layout, skipping')
IMGFIX
python -c "import imgaug; print('imgaug imports cleanly')"

# mmengine 0.10.7 calls torch.load() without weights_only=False. Since
# PyTorch 2.6 the default is weights_only=True, and loading our checkpoints
# (whose log history contains mmengine HistoryBuffer objects) then fails with
# UnpicklingError. The checkpoints are produced by this project, so relaxed
# loading is safe.
python - << 'MMFIX'
import re
import mmengine.runner.checkpoint as ckpt
src = ckpt.__file__
text = open(src).read()
if 'weights_only=False' in text:
    print('mmengine already patched, skipping')
else:
    new = re.sub(r'torch\.load\((.*?), map_location=map_location\)',
                 r'torch.load(\1, map_location=map_location, '
                 r'weights_only=False)', text)
    open(src, 'w').write(new)
    print('mmengine patched for PyTorch >= 2.6')
MMFIX

echo "Done. Next steps:"
echo "  1) Download the pretrained MonkeyOCRv2-AS backbone:"
echo "       hf download zenosai/MonkeyOCRv2-AS --local-dir $MMOCR_DIR/pretrained/monkeyocrv2_as"
echo "  2) Prepare the datasets (see README.md - Datasets)."
