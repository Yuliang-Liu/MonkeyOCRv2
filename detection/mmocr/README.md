
# MonkeyOCRv2 Detection

This repository provides the text detection experiments from the
[MonkeyOCRv2 paper](https://arxiv.org/abs/2607.11562). The visual encoder from
[MonkeyOCRv2-AS](https://huggingface.co/zenosai/MonkeyOCRv2-AS) (ViTAEv2-S,
21M parameters) is integrated into **DBNet** and **PSENet** scene text
detectors via [MMOCR](https://github.com/open-mmlab/mmocr). The four-stage
ViTAEv2 features (strides 4/8/16/32) are exposed as `res2`–`res5` and fed to
the standard FPNC / FPNF necks, so no change to the detection heads is
required.

Training and evaluation follow the official MMOCR protocols on Total-Text,
CTW1500, and ICDAR2015.

## Models and Results

For each detector, three visual backbones are compared under identical
settings: the original ImageNet-pretrained encoder, the text-specific
[oCLIP](https://github.com/bytedance/oclip) encoder, and MonkeyOCRv2.
MonkeyOCRv2 consistently improves F-score across all datasets and detector
architectures.

### Total-Text

| Method                  |    P |    R |        F |
| ----------------------- | ---: | ---: | -------: |
| DBNet (ResNet-50)       | 82.6 | 78.4 |     80.4 |
| DBNet + oCLIP           | 85.1 | 81.7 |     83.4 |
| **DBNet + MonkeyOCRv2** | 87.1 | 81.5 | **84.2** |

### CTW1500

| Method                   |    P |    R |        F |
| ------------------------ | ---: | ---: | -------: |
| PSENet (ResNet-50)       | 80.1 | 82.7 |     81.4 |
| PSENet + oCLIP           | 82.1 | 85.5 |     83.8 |
| **PSENet + MonkeyOCRv2** | 86.0 | 83.2 | **84.6** |

### ICDAR2015

| Method                   |    P |    R |        F |
| ------------------------ | ---: | ---: | -------: |
| PSENet (ResNet-50)       | 84.0 | 76.2 |     79.9 |
| PSENet + oCLIP           | 87.3 | 82.6 |     84.9 |
| **PSENet + MonkeyOCRv2** | 91.0 | 82.8 | **86.7** |
| DBNet (ResNet-50)        | 88.8 | 81.5 |     85.0 |
| DBNet + oCLIP            | 90.9 | 84.1 |     87.4 |
| **DBNet + MonkeyOCRv2**  | 91.3 | 86.7 | **88.9** |

### Checkpoints

Download the checkpoints from
[HB16888/MonkeyOCRv2\_det](https://huggingface.co/HB16888/MonkeyOCRv2_det)
(HuggingFace) or
[WangXinhan/MonkeyOCRv2\_det](https://modelscope.cn/models/WangXinhan/MonkeyOCRv2_det)
(ModelScope):

```bash
# run from this add-on directory; ./mmocr is the MMOCR root created by install.sh
# HuggingFace
hf download HB16888/MonkeyOCRv2_det --include "*.pth" --local-dir ./mmocr/model_weight
# ModelScope
modelscope download --model WangXinhan/MonkeyOCRv2_det --local_dir ./mmocr/model_weight
```

## Environment

The reproduced environment uses Python 3.11, PyTorch 2.9.0, CUDA 12.8,
MMEngine 0.10.7, MMCV 2.0.1, MMDetection 3.1.0, MMOCR 1.0.1, and
Transformers 4.57.1. Baselines and oCLIP models were trained on 2 GPUs; the
MonkeyOCRv2 PSENet models on 4 GPUs (NVIDIA GeForce RTX 3090).

## Installation

All commands in this section are run from this add-on directory
(`MonkeyOCRv2/detection/mmocr`).

### 1. Python, PyTorch, MMEngine, MMCV

```bash
conda create -y -n monkeyocrv2_det python=3.11
conda activate monkeyocrv2_det

pip install torch==2.9.0 torchvision==0.24.0 --index-url https://download.pytorch.org/whl/cu128
pip install mmengine==0.10.7

# MMCV 2.0.1 has no wheel for PyTorch 2.9, so build it from source
pip install "setuptools<81" wheel
git clone -b v2.0.1 --depth 1 https://github.com/open-mmlab/mmcv.git mmcv_src
sed -i 's/-std=c++14/-std=c++17/g' mmcv_src/setup.py
CUDA_HOME=/usr/local/cuda-12.8 MMCV_WITH_OPS=1 MAX_JOBS=16 pip install -e ./mmcv_src --no-build-isolation
python -c "from mmcv.ops import nms; import mmcv; print(mmcv.__version__)"
```

### 2. The add-on itself

This directory is an add-on on top of the official MMOCR v1.0.1. Run:

```bash
bash install.sh            # clones MMOCR v1.0.1 into ./mmocr and patches it
```

`install.sh` creates a nested MMOCR checkout, `./mmocr`, which is the **MMOCR
root** referred to throughout this README:

```text
MonkeyOCRv2/detection/mmocr/     # this add-on directory
├── install.sh
├── configs/  patch/  dataset_zoo/  tools/    # the add-on files, copied into ./mmocr
├── mmcv_src/                    # MMCV 2.0.1 source build (step 1)
└── mmocr/                       # <- MMOCR root, created by install.sh
    ├── configs/  tools/  dataset_zoo/
    ├── pretrained/monkeyocrv2_as/            # pretrained backbone
    ├── model_weight/                         # released checkpoints
    └── data/                                 # prepared datasets
```

## Pretrained Backbone

Download the MonkeyOCRv2-AS visual encoder before training or evaluation:

```bash
# run from this add-on directory
hf download zenosai/MonkeyOCRv2-AS --local-dir ./mmocr/pretrained/monkeyocrv2_as
```

## Datasets

```bash
cd mmocr                   # everything from here on runs in the MMOCR root
bash tools/dataset_converters/prepare_all_datasets.sh
```

This prepares `data/icdar2015`, `data/ctw1500`, and `data/totaltext` in the
MMOCR format. Note that some official CTW1500 download links are currently
unreliable; we also provide the prepared CTW1500 in MMOCR format at
[HB16888/CTW1500](https://huggingface.co/datasets/HB16888/CTW1500)
(HuggingFace) and
[WangXinhan/CTW1500](https://modelscope.cn/datasets/WangXinhan/CTW1500)
(ModelScope):

```bash
# HuggingFace
hf download HB16888/CTW1500 --repo-type dataset --local-dir ./ctw1500_dl
unzip ctw1500_dl/ctw1500_mmocr.zip -d data/
# ModelScope
modelscope download --dataset WangXinhan/CTW1500 --local_dir ./ctw1500_dl
unzip ctw1500_dl/ctw1500_mmocr.zip -d data/
```

Directory layout:

```text
data/ctw1500/
├── textdet_imgs/
│   ├── train/    # 1000 images
│   └── test/     # 500 images
├── textdet_train.json
└── textdet_test.json
```

## Training

All training and evaluation commands are run from the MMOCR root
(`detection/mmocr/mmocr`). Baselines and oCLIP models use 2 GPUs (batch size
16 in total); MonkeyOCRv2 PSENet models use 4 GPUs.

```bash
# ---------- DBNet on Total-Text ----------
bash tools/dist_train.sh configs/textdet/dbnet/dbnet_resnet50_1200e_totaltext_2gpu.py 2
bash tools/dist_train.sh configs/textdet/dbnet/dbnet_resnet50-oclip_1200e_totaltext_2gpu.py 2
bash tools/dist_train.sh configs/textdet/dbnet/dbnet_mkv2vitae_1200e_totaltext_2gpu_adamw.py 2

# ---------- PSENet on CTW1500 ----------
bash tools/dist_train.sh configs/textdet/psenet/psenet_resnet50_fpnf_600e_ctw1500_2gpu.py 2
bash tools/dist_train.sh configs/textdet/psenet/psenet_resnet50-oclip_fpnf_600e_ctw1500_2gpu.py 2
bash tools/dist_train.sh configs/textdet/psenet/psenet_mkv2vitae_fpnf_600e_ctw1500_4gpu_adamw.py 4

# ---------- PSENet on ICDAR2015 ----------
bash tools/dist_train.sh configs/textdet/psenet/psenet_resnet50_fpnf_600e_icdar2015_2gpu.py 2
bash tools/dist_train.sh configs/textdet/psenet/psenet_resnet50-oclip_fpnf_600e_icdar2015_2gpu.py 2
bash tools/dist_train.sh configs/textdet/psenet/psenet_mkv2vitae_fpnf_600e_icdar2015_4gpu_adamw.py 4

# ---------- DBNet on ICDAR2015 ----------
bash tools/dist_train.sh configs/textdet/dbnet/dbnet_resnet50_1200e_icdar2015_2gpu.py 2
bash tools/dist_train.sh configs/textdet/dbnet/dbnet_resnet50-oclip_1200e_icdar2015_2gpu.py 2
bash tools/dist_train.sh configs/textdet/dbnet/dbnet_mkv2vitae_1200e_icdar2015_2gpu_adamw.py 2
```

## Evaluation

```bash
# single GPU
python tools/test.py \
  configs/textdet/dbnet/dbnet_mkv2vitae_1200e_totaltext_2gpu_adamw.py \
  model_weight/dbnet_mkv2vitae_totaltext.pth

# multi GPU
bash tools/dist_test.sh \
  configs/textdet/psenet/psenet_mkv2vitae_fpnf_600e_ctw1500_4gpu_adamw.py \
  model_weight/psenet_mkv2vitae_ctw1500.pth 4
```

The evaluation prints `precision / recall / hmean` with the
`HmeanIOUMetric`, matching the tables above.

## Acknowledgements

This project builds on [MMOCR](https://github.com/open-mmlab/mmocr),
[DBNet](https://github.com/MhLiao/DB),
[PSENet](https://github.com/whai362/PSENet),
[oCLIP](https://github.com/bytedance/oclip), and
[MonkeyOCRv2](https://github.com/Yuliang-Liu/MonkeyOCRv2).
