# MonkeyOCRv2 Detection (DPText-DETR)

This repository provides the DPText-DETR text detection experiments from the
[MonkeyOCRv2 paper](https://arxiv.org/abs/2607.11562). The visual encoder from
[MonkeyOCRv2-AS](https://huggingface.co/zenosai/MonkeyOCRv2-AS) (ViTAEv2-S,
21M parameters) is integrated into
[DPText-DETR](https://github.com/ymy-k/DPText-DETR) as a drop-in detectron2
backbone. The last three ViTAEv2 stages (strides 8/16/32) are exposed as
`res3`–`res5` and feed the standard deformable-DETR input projections, so the
transformer encoder/decoder and the detection head are unchanged.

Training and evaluation follow the official DPText-DETR protocols on
Total-Text, CTW1500, ICDAR19-ArT, Rotated Total-Text and Inverse-Text.

## Models and Results

For each benchmark, three visual backbones are compared under identical
settings: the original ImageNet-pretrained ResNet-50, the text-specific
[oCLIP](https://github.com/bytedance/oclip) ResNet-50, and MonkeyOCRv2.
MonkeyOCRv2 consistently improves F-score across all five benchmarks.

All models are trained **directly on the target dataset** (no SynthText/MLT
pre-training), for 200k iterations with a total batch size of 8, using the
positional label form and the rotated training images released with
DPText-DETR (`*_poly_train_rotate_pos`).

### Total-Text

| Method                        |    P |    R |        F |
| ----------------------------- | ---: | ---: | -------: |
| DPText-DETR (ResNet-50)       | 89.6 | 82.8 |     86.1 |
| DPText-DETR + oCLIP           | 87.1 | 84.5 |     85.7 |
| **DPText-DETR + MonkeyOCRv2** | 90.9 | 86.7 | **88.8** |

### CTW1500

| Method                        |    P |    R |        F |
| ----------------------------- | ---: | ---: | -------: |
| DPText-DETR (ResNet-50)       | 89.7 | 82.1 |     85.7 |
| DPText-DETR + oCLIP           | 86.3 | 82.7 |     84.5 |
| **DPText-DETR + MonkeyOCRv2** | 89.6 | 88.1 | **88.9** |

### ICDAR19-ArT

| Method                        |    P |    R |        F |
| ----------------------------- | ---: | ---: | -------: |
| DPText-DETR (ResNet-50)       | 84.3 | 67.5 |     75.0 |
| DPText-DETR + oCLIP           | 75.1 | 62.0 |     67.9 |
| **DPText-DETR + MonkeyOCRv2** | 85.8 | 71.7 | **78.1** |

### Rotated Total-Text

| Method                        |    P |    R |        F |
| ----------------------------- | ---: | ---: | -------: |
| DPText-DETR (ResNet-50)       | 89.4 | 79.8 |     84.3 |
| DPText-DETR + oCLIP           | 87.2 | 80.8 |     83.9 |
| **DPText-DETR + MonkeyOCRv2** | 89.7 | 84.4 | **86.9** |

### Inverse-Text

| Method                        |    P |    R |        F |
| ----------------------------- | ---: | ---: | -------: |
| DPText-DETR (ResNet-50)       | 92.1 | 81.3 |     86.4 |
| DPText-DETR + oCLIP           | 90.2 | 82.1 |     85.9 |
| **DPText-DETR + MonkeyOCRv2** | 91.8 | 85.4 | **88.5** |

Rotated Total-Text and Inverse-Text are **test-only** benchmarks: they reuse
the Total-Text model above and only change `DATASETS.TEST`.

### Checkpoints

Download the checkpoints from
[HB16888/MonkeyOCRv2\_det\_dptext](https://huggingface.co/HB16888/MonkeyOCRv2_det_dptext)
(HuggingFace) or
[WangXinhan/MonkeyOCRv2\_det\_dptext](https://modelscope.cn/models/WangXinhan/MonkeyOCRv2_det_dptext)
(ModelScope):

```bash
# run from this add-on directory; ./DPText-DETR is the repository root created
# by install.sh
# HuggingFace
hf download HB16888/MonkeyOCRv2_det_dptext --include "*.pth" --local-dir ./DPText-DETR/model_weight
# ModelScope
modelscope download --model WangXinhan/MonkeyOCRv2_det_dptext --local_dir ./DPText-DETR/model_weight
```

## Environment

The reproduced environment uses Python 3.11, PyTorch 2.9.0, CUDA 12.8,
torchvision 0.24.0, detectron2 0.6, NumPy 2.4.4, Transformers 4.57.1 and
safetensors 0.7.0. The oCLIP baseline additionally needs MMOCR 1.0.1
(MMEngine 0.10.7, MMCV 2.0.1, MMDet 3.1.0). All models were trained on 8 GPUs
(NVIDIA GeForce RTX 3090) with `SOLVER.IMS_PER_BATCH: 8` for 200k iterations.

## Installation

This directory is an add-on on top of the official DPText-DETR release. Run it
from this add-on directory (`MonkeyOCRv2/detection/DPText-DETR`):

```bash
bash install.sh            # clones DPText-DETR into ./DPText-DETR and patches it
```

`install.sh` creates a nested checkout, `./DPText-DETR`, which is the
**DPText-DETR root** referred to throughout this README:

```text
MonkeyOCRv2/detection/DPText-DETR/     # this add-on directory
├── install.sh
├── configs/  patch/  tools/           # the add-on files, copied into ./DPText-DETR
└── DPText-DETR/                       # <- DPText-DETR root, created by install.sh
    ├── adet/  configs/  tools/
    ├── pretrained/monkeyocrv2_as/     # MonkeyOCRv2-AS visual encoder
    ├── ckpts/                         # ResNet-50 / oCLIP init weights
    ├── model_weight/                  # released checkpoints
    ├── datasets/                      # the benchmarks
    └── output/                        # training / evaluation output
```

## Pretrained Backbones

```bash
cd DPText-DETR             # the DPText-DETR root created by install.sh

# MonkeyOCRv2-AS visual encoder (for the MonkeyOCRv2 rows)
hf download zenosai/MonkeyOCRv2-AS --local-dir ./pretrained/monkeyocrv2_as

# ImageNet ResNet-50 (for the baseline rows) - from the official DPText-DETR /
# AdelaiDet instructions
mkdir -p ckpts
wget -O ckpts/R-50.pkl https://dl.fbaipublicfiles.com/detectron2/ImageNetPretrained/MSRA/R-50.pkl

# oCLIP ResNet-50 (for the oCLIP rows)
wget -O ckpts/resnet50-oclip-7ba0c533.pth \
  https://download.openmmlab.com/mmocr/backbone/resnet50-oclip-7ba0c533.pth
```

## Datasets

Download Total-Text (including rotated images), CTW1500 (including rotated
images), ICDAR19-ArT (including rotated images), Inverse-Text, the polygon
json files and the evaluation ground-truths from the official
[DPText-DETR data preparation](https://github.com/ymy-k/DPText-DETR#data-preparation)
links, and organize them under the DPText-DETR root as:

```text
datasets/
├── totaltext/
│   ├── train_images_rotate/
│   ├── test_images_rotate/
│   ├── train_poly_rotate_pos.json
│   ├── test_poly.json
│   └── test_poly_rotate.json
├── ctw1500/
│   ├── train_images_rotate/
│   ├── test_images/
│   ├── train_poly_rotate_pos.json
│   └── test_poly.json
├── art/
│   ├── train_images_rotate/
│   ├── test_images/
│   ├── train_poly_rotate_pos.json
│   └── test_poly.json
├── inversetext/
│   ├── test_images/
│   └── test_poly.json
└── evaluation/
    ├── gt_totaltext.zip
    ├── gt_totaltext_rotate.zip
    ├── gt_ctw1500.zip
    └── gt_inversetext.zip
```

## Training

All training and evaluation commands are run from the DPText-DETR root
(`detection/DPText-DETR/DPText-DETR`), on 8 GPUs.

```bash
# ---------- Total-Text (also used for Rot.Total-Text and Inverse-Text) ----------
python tools/train_net.py --config-file configs/DPText_DETR/TotalText_Direct_Rotate/R_50_poly.yaml             --num-gpus 8
python tools/train_net.py --config-file configs/DPText_DETR/TotalText_Direct_Rotate/R_50_oclip_poly_lr1e4.yaml --num-gpus 8
python tools/train_net.py --config-file configs/DPText_DETR/TotalText_Direct_Rotate/mkv2vitae_align.yaml       --num-gpus 8

# ---------- CTW1500 ----------
python tools/train_net.py --config-file configs/DPText_DETR/CTW_Rotate/R_50_poly.yaml             --num-gpus 8
python tools/train_net.py --config-file configs/DPText_DETR/CTW_Rotate/R_50_oclip_poly_lr1e4.yaml --num-gpus 8
python tools/train_net.py --config-file configs/DPText_DETR/CTW_Rotate/mkv2vitae_align.yaml       --num-gpus 8

# ---------- ICDAR19-ArT ----------
python tools/train_net.py --config-file configs/DPText_DETR/ArT_Rotate/R_50_poly.yaml             --num-gpus 8
python tools/train_net.py --config-file configs/DPText_DETR/ArT_Rotate/R_50_oclip_poly_lr1e4.yaml --num-gpus 8
python tools/train_net.py --config-file configs/DPText_DETR/ArT_Rotate/mkv2vitae_align.yaml       --num-gpus 8
```

## Evaluation

Each config already carries the `MODEL.TRANSFORMER.INFERENCE_TH_TEST` value
that reproduces the corresponding row of the tables above, so evaluating on
the dataset a model was trained on needs no extra flags:

```bash
# Total-Text
python tools/train_net.py --num-gpus 8 --eval-only \
  --config-file configs/DPText_DETR/TotalText_Direct_Rotate/mkv2vitae_align.yaml \
  MODEL.WEIGHTS model_weight/dptext_mkv2vitae_totaltext.pth

# CTW1500
python tools/train_net.py --num-gpus 8 --eval-only \
  --config-file configs/DPText_DETR/CTW_Rotate/mkv2vitae_align.yaml \
  MODEL.WEIGHTS model_weight/dptext_mkv2vitae_ctw1500.pth
```

Evaluation prints `precision / recall / hmean` on the `copypaste:` line,
matching the tables above.

### Rotated Total-Text and Inverse-Text

These reuse the Total-Text checkpoints and override the test set and the
threshold:

```bash
# Rotated Total-Text
python tools/train_net.py --num-gpus 8 --eval-only \
  --config-file configs/DPText_DETR/TotalText_Direct_Rotate/mkv2vitae_align.yaml \
  MODEL.WEIGHTS model_weight/dptext_mkv2vitae_totaltext.pth \
  MODEL.TRANSFORMER.INFERENCE_TH_TEST 0.395 \
  DATASETS.TEST '("totaltext_poly_test_rotate",)'

# Inverse-Text
python tools/train_net.py --num-gpus 8 --eval-only \
  --config-file configs/DPText_DETR/TotalText_Direct_Rotate/mkv2vitae_align.yaml \
  MODEL.WEIGHTS model_weight/dptext_mkv2vitae_totaltext.pth \
  MODEL.TRANSFORMER.INFERENCE_TH_TEST 0.37 \
  DATASETS.TEST '("inversetext_test",)'
```

The full set of thresholds used for the tables:

| Backbone    | Total-Text | Rot.Total-Text | Inverse-Text | CTW1500 |   ArT |
| ----------- | ---------: | -------------: | -----------: | ------: | ----: |
| ResNet-50   |       0.37 |          0.415 |         0.45 |   0.495 | 0.375 |
| oCLIP       |       0.34 |           0.34 |         0.37 |   0.365 |  0.35 |
| MonkeyOCRv2 |      0.405 |          0.395 |         0.37 |   0.375 | 0.355 |

`tools/search_th.py` sweeps `INFERENCE_TH_TEST` for a trained model and
reports the best F-score:

```bash
python tools/search_th.py \
  --output-dir output/mkv2vitae_align/totaltext/direct_rotate \
  --test-dataset totaltext_poly_test --start 0.1 --end 0.5 --num-gpus 8
```

### ICDAR19-ArT

ArT has no public test ground-truth. Evaluating an ArT config writes
`<OUTPUT_DIR>/inference/art_submit.json`, which has to be uploaded to the
[ICDAR19-ArT evaluation server](https://rrc.cvc.uab.es/?ch=14) to obtain the
P / R / F numbers reported above:

```bash
python tools/train_net.py --num-gpus 8 --eval-only \
  --config-file configs/DPText_DETR/ArT_Rotate/mkv2vitae_align.yaml \
  MODEL.WEIGHTS model_weight/dptext_mkv2vitae_art.pth
```

## Acknowledgements

This project builds on [DPText-DETR](https://github.com/ymy-k/DPText-DETR),
[AdelaiDet](https://github.com/aim-uofa/AdelaiDet),
[detectron2](https://github.com/facebookresearch/detectron2),
[MMOCR](https://github.com/open-mmlab/mmocr),
[oCLIP](https://github.com/bytedance/oclip), and
[MonkeyOCRv2](https://github.com/Yuliang-Liu/MonkeyOCRv2).

## License

The DPText-DETR / AdelaiDet sources this add-on patches are released for
**non-commercial use only** (see [LICENSE](LICENSE)); the same restriction
applies to this directory and to the released checkpoints.
