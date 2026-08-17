# MonkeyOCRv2 Recognition

MonkeyOCRv2 Recognition combines the visual encoder from
[MonkeyOCRv2-S](https://huggingface.co/zenosai/MonkeyOCRv2-S) with PARSeq and
CRNN decoders for English and Chinese scene text recognition.

## Models and Results
Results are reported by
[SVTRv2](https://github.com/Topdu/OpenOCR/tree/main/configs/rec/svtrv2), except
for the MonkeyOCRv2-S models.

### Comprehensive Benchmarks

Integrating MonkeyOCRv2-S into CRNN and PARSeq consistently improves results
on Union14M-Benchmark, Chinese Benchmark, and Occluded Scene Text (OST).
Overall is the average of the three benchmark averages.

| Model | Overall | U14M Avg | Artistic | Contextless | Curve | General | Multi Oriented | Multi Words | Saliency | Chinese Avg | Scene | Web | Document | Handwriting | OST |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ABINet | 73.7 | 75.7 | 71.7 | 74.7 | 80.4 | 79.8 | 69.0 | 76.8 | 77.6 | 70.3 | 66.6 | 63.2 | 98.2 | 53.1 | 75.0 |
| MAERec | 81.6 | 85.2 | 79.0 | 84.2 | 89.1 | 84.6 | 87.1 | 85.9 | 86.3 | 83.1 | 84.4 | 83.0 | **99.5** | 65.6 | 76.4 |
| CPPD | 81.1 | 81.9 | 76.5 | 82.9 | 86.2 | 83.5 | 78.7 | 81.9 | 83.5 | 81.7 | 82.7 | 82.4 | 99.4 | 62.3 | 79.6 |
| IGTR-AR | 81.0 | 84.9 | 77.0 | 82.4 | 90.4 | 84.4 | 91.2 | 84.0 | 84.7 | 81.7 | 82.0 | 81.7 | **99.5** | 63.8 | 76.3 |
| SMTR | 80.4 | 85.0 | 76.8 | 83.9 | 89.1 | 83.7 | 87.7 | **89.3** | 84.6 | 82.7 | 83.4 | 83.0 | 99.3 | 65.1 | 73.5 |
| SVTRv2 | 83.1 | 86.1 | **79.3** | 86.1 | 90.6 | 85.1 | 89.0 | 86.7 | 86.2 | 83.3 | 83.5 | **83.3** | **99.5** | 67.0 | 80.0 |
| CRNN (ResNet) | 58.7 | 49.2 | 51.2 | 62.3 | 48.1 | 68.2 | 13.0 | 60.4 | 41.4 | 68.8 | 63.8 | 68.2 | 97.0 | 46.1 | 58.0 |
| CRNN (MonkeyOCRv2-S) | 67.3 | 65.2 | 63.7 | 73.0 | 71.1 | 74.5 | 28.6 | 72.1 | 73.4 | 74.2 | 73.0 | 74.9 | 96.9 | 51.8 | 62.4 |
| PARSeq (ViT) | 82.2 | 84.3 | 76.5 | 83.4 | 87.6 | 84.9 | 88.8 | 84.3 | 84.4 | 82.4 | 84.2 | 82.8 | **99.5** | 63.0 | 79.9 |
| **PARSeq (MonkeyOCRv2-S)** | **84.3** | **87.6** | 78.6 | **86.4** | **92.1** | **85.4** | **93.9** | 88.7 | **87.7** | **83.7** | **84.6** | 83.2 | **99.5** | **67.3** | **81.5** |

### Common Benchmarks

| Model | Avg | IIIT5k | SVT | IC13 | IC15 | SVTP | CUTE80 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ABINet | 95.8 | 98.5 | 98.1 | 97.7 | 90.1 | 94.1 | 96.5 |
| MAERec | 96.4 | **99.2** | 97.8 | 98.2 | 90.4 | 94.3 | 98.3 |
| CPPD | 96.4 | 99.0 | 97.8 | 98.2 | 90.4 | 94.0 | **99.0** |
| IGTR-AR | 96.5 | 98.7 | **98.4** | 98.1 | 90.5 | 94.9 | 98.3 |
| SMTR | 95.9 | 99.0 | 97.4 | 98.3 | 90.1 | 92.7 | 97.9 |
| SVTRv2 | 96.6 | **99.2** | 98.0 | **98.7** | **91.1** | 93.5 | **99.0** |
| CRNN (ResNet) | 90.2 | 95.8 | 91.8 | 94.6 | 84.9 | 83.1 | 91.0 |
| CRNN (MonkeyOCRv2-S) | 92.5 | 97.4 | 94.3 | 96.5 | 86.8 | 87.1 | 93.1 |
| PARSeq (ViT) | 96.4 | 98.9 | 98.1 | 98.4 | 90.1 | 94.3 | 98.6 |
| **PARSeq (MonkeyOCRv2-S)** | **96.8** | **99.2** | 98.0 | 98.5 | 90.4 | **96.1** | 98.6 |

Download the checkpoints from
[echo840/MonkeyOCRv2_rec](https://huggingface.co/echo840/MonkeyOCRv2_rec):

```bash
hf download echo840/MonkeyOCRv2_rec --include "*.ckpt" --local-dir ./model_weight
```

## Environment

The reproduced environment uses Python 3.10, PyTorch 2.10.0, CUDA 12.8, and a single NVIDIA H800 GPU.

```bash
conda create -n monkeyocrv2_rec python=3.10 pip
conda activate monkeyocrv2_rec
pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
pip install -e .
```

FlashAttention is optional. The implementation automatically uses PyTorch SDPA when FlashAttention is unavailable.

## Pretrained Encoder

Download the MonkeyOCRv2-S visual encoder before training, evaluation, or
inference:

```bash
hf download zenosai/MonkeyOCRv2-S --local-dir ./pretrained/monkeyocr_vit
```

## Datasets

### English

The English training and evaluation LMDBs follow the
[OpenOCR SVTRv2 data protocol](https://github.com/Topdu/OpenOCR/blob/main/docs/svtrv2.md#downloading-datasets).
Download Union14M-L-Filtered, Common Benchmarks, Union14M-Benchmark, and OST
from [topdu/OpenOCR-Data](https://huggingface.co/datasets/topdu/OpenOCR-Data):

```bash
hf download topdu/OpenOCR-Data --repo-type dataset \
  --include "Union14M-L-LMDB-Filtered/*" "evaluation/*" "u14m/*" "OST/*" \
  --local-dir ./data/openocr
python scripts/prepare_english_data.py
```

### Chinese
Follow the official
[Benchmarking Chinese Text Recognition](https://github.com/FudanVI/benchmarking-chinese-text-recognition#download)
instructions. Scene, web, and document LMDBs are available from its official
Google Drive; handwriting data must be prepared from SCUT-HCCDoc using the
official split.

### Directory Layout

Each leaf directory below is an LMDB containing `data.mdb` and `lock.mdb`.

```text
./data/
├── english/
│   ├── train/
│   │   ├── filter_train_easy/
│   │   ├── filter_train_medium/
│   │   ├── filter_train_hard/
│   │   ├── filter_train_normal/
│   │   └── filter_train_challenging/
│   └── test/
│       ├── IIIT5k/ SVT/ SVTP/ IC13_857/ IC15_1811/ CUTE80/
│       ├── u14m/
│       │   ├── artistic/ contextless/ curve/ general/
│       │   └── multi_oriented/ multi_words/ salient/
│       └── ost/
│           ├── heavy/
│           └── weak/
└── chinese/
    ├── train/
    │   ├── scene_train/ web_train/ document_train/ handwriting_train/
    ├── val/
    │   ├── scene_val/ web_val/ document_val/ handwriting_val/
    └── test/
        └── scene_test/ web_test/ document_test/ handwriting_test/
```

## Training

Training uses two stages: the visual encoder is frozen in stage 1 and all parameters are optimized in stage 2. The total number of training epochs is kept consistent with SVTRv2.



```bash
# PARSeq English
bash run/train_english_stage1.sh
bash run/train_english_stage2.sh outputs/english_stage1/<stage1_run>

# PARSeq Chinese
bash run/train_chinese_stage1.sh
bash run/train_chinese_stage2.sh outputs/chinese_stage1/<stage1_run>

# CRNN English
bash run/train_crnn_english_stage1.sh
bash run/train_crnn_english_stage2.sh outputs/crnn_english_stage1/<stage1_run>

# CRNN Chinese
bash run/train_crnn_chinese_stage1.sh
bash run/train_crnn_chinese_stage2.sh outputs/crnn_chinese_stage1/<stage1_run>
```

Checkpoints and TensorBoard logs are written under `outputs/`.

## Evaluation
Result JSON files are written to `eval_results/`.

```bash
python scripts/eval_english.py model_weight/parseq_en.ckpt \
  --json_out eval_results/parseq_en.json

python scripts/eval_chinese.py model_weight/parseq_zh.ckpt \
  --json_out eval_results/parseq_zh.json

python scripts/eval_crnn.py
```

## Inference

```bash
# PARSeq English
python scripts/infer_english.py ./model_weight/parseq_en.ckpt \
  --images ./demo_images/ic13_word_256.png

# PARSeq Chinese
python scripts/infer_chinese.py ./model_weight/parseq_zh.ckpt \
  --images ./demo_images/chinese_scene_test_000001.png

# CRNN English
python scripts/infer_english.py ./model_weight/crnn_en.ckpt \
  --images ./demo_images/ic13_word_256.png

# CRNN Chinese
python scripts/infer_chinese.py ./model_weight/crnn_zh.ckpt \
  --images ./demo_images/chinese_scene_test_000001.png
```


## Demo

```bash
python scripts/demo_gradio.py
```

Open `http://127.0.0.1:7860`.

## Acknowledgements

This project builds on [PARSeq](https://github.com/baudm/parseq),
[MonkeyOCRv2](https://github.com/Yuliang-Liu/MonkeyOCRv2),
[OpenOCR](https://github.com/Topdu/OpenOCR), and the
[Chinese text recognition benchmark](https://github.com/FudanVI/benchmarking-chinese-text-recognition).
