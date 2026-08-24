#!/bin/bash
# Install the MonkeyOCRv2 text-detection add-on on top of the official
# DPText-DETR release.
#
# Usage:
#   conda create -n dptext python=3.11 -y && conda activate dptext
#   bash install.sh [TARGET_DPTEXT_DIR]      # default: ./DPText-DETR
#
# Environment variables:
#   SKIP_DEPS=1     skip the PyTorch / detectron2 / dependency install and only
#                   clone, patch and build (use when the env is already set up)
#   SKIP_MMOCR=1    skip MMOCR, which is only needed for the oCLIP baseline
#
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DPTEXT_DIR="${1:-$SCRIPT_DIR/DPText-DETR}"
DPTEXT_GIT="${DPTEXT_GIT:-https://github.com/ymy-k/DPText-DETR.git}"
DPTEXT_COMMIT="${DPTEXT_COMMIT:-c62baaed9da69b58280b2756720eb37c19a91058}"
DETECTRON2_COMMIT="${DETECTRON2_COMMIT:-b599f139756bd3646a26a909caf86a1a159e53a7}"

if [ -z "$SKIP_DEPS" ]; then
    echo "==> Installing the CUDA toolkit (provides nvcc for the CUDA extensions) ..."
    # detectron2 and the AdelaiDet deformable-attention kernels are compiled from
    # source, so a full CUDA toolkit is required, not just the runtime shipped
    # with the PyTorch wheels. gxx_linux-64 provides a host compiler new enough
    # for the PyTorch 2.9 headers.
    if command -v conda > /dev/null; then
        conda install -y -c conda-forge \
            cuda-nvcc=12.9 cuda-cudart-dev=12.9 cuda-libraries-dev=12.9 gxx_linux-64
    else
        echo "conda not found - make sure nvcc is on PATH and CUDA_HOME is set."
    fi

    if [ -z "$CUDA_HOME" ] && [ -n "$CONDA_PREFIX" ] && [ -x "$CONDA_PREFIX/bin/nvcc" ]; then
        export CUDA_HOME="$CONDA_PREFIX"
    fi

    echo "==> Installing PyTorch 2.9.0 (cu128) ..."
    pip install torch==2.9.0 torchvision==0.24.0 --index-url https://download.pytorch.org/whl/cu128

    echo "==> Installing DPText-DETR runtime dependencies ..."
    # Pinned to the versions the reported numbers were produced with. In
    # particular opencv >= 5 and transformers >= 5 change the results slightly.
    pip install \
        "numpy==2.4.4" "opencv-python==4.13.0.92" "pillow==12.1.1" \
        "scipy==1.17.1" "timm==1.0.26" "shapely==2.1.2" \
        "albumentations==2.0.8" "Polygon3==3.0.9.1" "rapidfuzz==3.14.5" \
        "setuptools==80.10.2"
    # Required by the MonkeyOCRv2-ViTAE backbone (a HuggingFace custom-code model).
    pip install "transformers==4.57.1" "safetensors==0.7.0"

    # The reported numbers were produced with Pillow linked against libjpeg 9.
    # The PyPI Pillow wheel bundles libjpeg-turbo instead, which decodes JPEGs
    # slightly differently and shifts the metrics by up to 0.2 points. Install
    # the conda build so the decoder matches.
    if command -v conda > /dev/null; then
        conda install -y "pillow=12.1.1" "jpeg=9f"
    fi

    echo "==> Installing detectron2 ..."
    # detectron2's setup.py imports torch, so build isolation has to be off.
    pip install --no-build-isolation "git+https://github.com/facebookresearch/detectron2.git@${DETECTRON2_COMMIT}"

    if [ -z "$SKIP_MMOCR" ]; then
        echo "==> Installing MMOCR (only needed for the oCLIP baseline) ..."
        pip install --no-build-isolation "mmengine==0.10.7"
        # mmcv 2.0.1 hardcodes -std=c++14, which does not compile against the
        # PyTorch 2.9 headers, so build a C++17-patched copy from the sdist.
        MMCV_SRC="$(mktemp -d)"
        MMCV_URL="$(python -c "import json,urllib.request as u; d=json.load(u.urlopen('https://pypi.org/pypi/mmcv/2.0.1/json')); print([x['url'] for x in d['urls'] if x['packagetype']=='sdist'][0])")"
        curl -sL -o "$MMCV_SRC/mmcv.tar.gz" "$MMCV_URL"
        tar xzf "$MMCV_SRC/mmcv.tar.gz" -C "$MMCV_SRC"
        sed -i 's/-std=c++14/-std=c++17/g' "$MMCV_SRC/mmcv-2.0.1/setup.py"
        MMCV_WITH_OPS=1 pip install --no-build-isolation "$MMCV_SRC/mmcv-2.0.1"
        rm -rf "$MMCV_SRC"
        pip install --no-build-isolation "mmdet==3.1.0" "mmocr==1.0.1"
        # mmocr pulls in a tifffile that requires Python >= 3.12 syntax.
        pip install "tifffile==2026.3.3"
    fi
fi

if [ ! -d "$DPTEXT_DIR/.git" ]; then
    echo "==> Cloning DPText-DETR ($DPTEXT_COMMIT) from $DPTEXT_GIT ..."
    git clone "$DPTEXT_GIT" "$DPTEXT_DIR"
    git -C "$DPTEXT_DIR" checkout "$DPTEXT_COMMIT"
else
    echo "==> Using existing DPText-DETR checkout at $DPTEXT_DIR"
fi

echo "==> Copying add-on files ..."
cp -r "$SCRIPT_DIR/configs/." "$DPTEXT_DIR/configs/"
cp -r "$SCRIPT_DIR/patch/adet/." "$DPTEXT_DIR/adet/"
cp -r "$SCRIPT_DIR/tools/." "$DPTEXT_DIR/tools/"

echo "==> Building the AdelaiDet CUDA extensions ..."
cd "$DPTEXT_DIR"
if [ -z "$CUDA_HOME" ] && [ -n "$CONDA_PREFIX" ] && [ -x "$CONDA_PREFIX/bin/nvcc" ]; then
    export CUDA_HOME="$CONDA_PREFIX"
fi
# Editable install. Build isolation has to be off: setup.py imports torch.
pip install --no-build-isolation -e .

echo
echo "Done. Next steps:"
echo "  1) Download the pretrained MonkeyOCRv2-AS backbone:"
echo "       hf download zenosai/MonkeyOCRv2-AS --local-dir $DPTEXT_DIR/pretrained/monkeyocrv2_as"
echo "  2) Download the ResNet-50 / oCLIP init weights into $DPTEXT_DIR/ckpts (see README.md)."
echo "  3) Prepare the datasets (see README.md - Datasets)."
