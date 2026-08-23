# MonkeyOCRv2 Formula Recognition

This folder provides the training, inference, and evaluation code for
MonkeyOCRv2 formula recognition.  The released model is built on UniMERNet-T
with the MonkeyOCRv2-S visual encoder.

## Method

We replace UniMERNet-T's original Swin visual encoder with MonkeyOCRv2-S and
keep the MBart decoder unchanged.  Because MonkeyOCRv2-S outputs 384-d visual
tokens while the decoder uses 512-d hidden states, a learnable linear projection
maps encoder features from 384 to 512 before decoder cross-attention.

Following UniMERNet, we use the public fine-tuning and CDM evaluation pipeline.
The UniMERNet-T (Swin) baseline uses a Swin encoder pretrained on 16M in-house
data; this closed pre-training stage is not included in our release.

## Models and Results

Results are reported on OmniDocBench 1.6, MathWriting, and UniMER-Test.  CDM is
the rendered-formula matching score, and ExpRate is expression-level exact
rendering rate.

| Model | Params | Overall CDM | Overall ExpRate | OmniDocBench 1.6 CDM | OmniDocBench 1.6 ExpRate | MathWriting CDM | MathWriting ExpRate | SPE CDM | SPE ExpRate | CPE CDM | CPE ExpRate | HWE CDM | HWE ExpRate | SCE CDM | SCE ExpRate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Pix2tex | 25.5M | 53.8 | 23.3 | 69.4 | 27.0 | 0.4 | 0.0 | 96.2 | 72.4 | 64.9 | 7.1 | 24.5 | 0.6 | 67.6 | 32.8 |
| Texify | 312M | 67.3 | 40.4 | 76.5 | 46.4 | 26.6 | 2.0 | 98.5 | 91.0 | 70.4 | 28.2 | 52.7 | 23.6 | 79.3 | 51.3 |
| UniMERNet-B | 325M | 89.5 | 64.5 | 90.4 | 59.5 | 63.8 | 12.3 | 99.1 | 93.3 | 96.0 | **80.5** | 94.0 | 64.3 | 93.7 | 77.0 |
| UniMERNet-S | 202M | 89.8 | 64.0 | 90.1 | 59.1 | 65.9 | 12.7 | 99.1 | 93.4 | 95.9 | 77.7 | 93.7 | 63.9 | **94.1** | 76.9 |
| UniMERNet-T (Swin) | 107M | 89.4 | 61.8 | 89.9 | 57.2 | 65.6 | 12.9 | 99.1 | 92.3 | 94.9 | 69.9 | 93.3 | 61.9 | 93.8 | 76.6 |
| **UniMERNet-T (MonkeyOCRv2-S)** | 110M | **90.9** | **66.4** | **90.8** | **61.1** | **70.8** | **16.2** | **99.2** | **93.8** | **96.1** | 79.2 | **94.3** | **69.5** | 94.0 | **78.6** |

Replacing Swin with MonkeyOCRv2-S improves UniMERNet-T on all three benchmarks,
with clear gains on OmniDocBench 1.6, MathWriting, CPE, and HWE, while keeping
the model compact at 110M parameters.

## Environment

The reproduced environment uses Python 3.10, PyTorch 2.x, CUDA, and 4 GPUs for
training.  CDM additionally requires a working LaTeX toolchain, ImageMagick,
Node.js, and the Python packages in `cdm_local/requirements.txt`.

```bash
cd formula
conda create -n monkeyocrv2_formula python=3.10 pip
conda activate monkeyocrv2_formula
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install -r cdm_local/requirements.txt  # required for CDM evaluation
pip install -e .
```

## Model Files

The code expects this layout:

```text
formula/
+-- models/
|   +-- unimernet_tiny/
|   +-- monkeyocrv2_vit/
+-- model_weight/
    +-- monkeyocrv2_s_formula.pth
```

Download the public base components with:

```bash
bash run/download_models.sh
```

The script downloads the released checkpoint from
[`zhd36/monkeyocrv2_s_formula`](https://huggingface.co/zhd36/monkeyocrv2_s_formula)
to `model_weight/monkeyocrv2_s_formula.pth`.  Set `FORMULA_CKPT_REPO` or
`FORMULA_CKPT_FILE` to override the repository or filename.

## Datasets

UniMER data follows the original UniMERNet layout:

```text
data/
+-- UniMER1M/
|   +-- images/*.png
|   +-- train.txt
+-- UniMER-Test/
|   +-- cpe/*.png  hwe/*.png  sce/*.png  spe/*.png
|   +-- cpe.txt    hwe.txt    sce.txt    spe.txt
+-- UniMER1M_HME100K_merged/
    +-- images/*.png
    +-- train.txt
```

Download UniMER files:

```bash
bash run/download_unimer_data.sh
```

The released training recipe uses the merged UniMER-1M + HME100K training set.  After
placing the HME100K archive locally, build the merged set with:

```bash
HME100K_ARCHIVE=/path/to/HME100K.zip bash run/prepare_train_data.sh
```

For MathWriting:

```bash
bash run/download_mathwriting.sh
```

For OmniDocBench, place the official JSON and page images under:

```text
data/omnidocbench/OmniDocBench.json
data/omnidocbench/images/
```

or pass `ODB_JSON=/path/to/OmniDocBench.json` and
`ODB_IMAGE_ROOT=/path/to/images` to the evaluation script.

```bash
bash run/download_omnidocbench.sh
```

## Training

Training uses two stages.  Stage 1 freezes the MonkeyOCRv2-S visual encoder for
50k iterations; stage 2 unfreezes all parameters and trains for 250k iterations.

```bash
# 4-GPU training by default
bash run/train_stage1.sh
bash run/train_stage2.sh
```

Common overrides:

```bash
GPUS=0,1 NPROC=2 bash run/train_stage1.sh run.batch_size_train=8
STAGE1_CKPT=outputs/monkeyocrv2_s_stage1_freeze50k/<run>/checkpoint_latest.pth bash run/train_stage2.sh
```

Checkpoints and logs are written under `outputs/` and `logs_train/`.

## Inference

```bash
python scripts/infer.py --images ../images_test/formula.png

python scripts/infer.py \
  --image-dir /path/to/formula_images \
  --output eval_results/demo_predictions.jsonl
```

## Evaluation

UniMER-Test:

```bash
bash run/eval_unimer_test.sh
```

MathWriting:

```bash
bash run/eval_mathwriting.sh
```

OmniDocBench:

```bash
bash run/eval_omnidocbench.sh
```

Run all benchmarks:

```bash
bash run/eval_all.sh
```

Useful evaluation environment variables:

```bash
CHECKPOINT=model_weight/monkeyocrv2_s_formula.pth
BATCH_SIZE=128
NUM_WORKERS=8
RUN_CDM=1
POOLS=64
```

To reproduce official OmniDocBench Edit/BLEU outputs in addition to CDM, install
the OmniDocBench evaluator and run:

```bash
RUN_OFFICIAL=1 ODB_EVAL_REPO=/path/to/OmniDocBench bash run/eval_omnidocbench.sh
```

### Reproducing Table Metrics

The reported CDM / ExpRate numbers are sensitive to the CDM rendering environment.
For table-level reproduction, use the same CDM pipeline:

- TeX Live 2025, not arbitrary system TeX.
- Original CDM color rendering based on `\mathcolor[RGB]`.
- Fixed PDF-to-image backend used by the released evaluation.
- For OmniDocBench, run the official OmniDocBench evaluator first and use its exported `*_formula.json` as
CDM input.


## Acknowledgements

This project builds on [UniMERNet](https://github.com/opendatalab/UniMERNet),
its [CDM](https://github.com/opendatalab/UniMERNet/tree/main/cdm) evaluation
toolkit, and the MBart decoder implementation from
[Transformers](https://github.com/huggingface/transformers).  We also thank the
maintainers of [OmniDocBench](https://github.com/opendatalab/OmniDocBench),
[MathWriting](https://github.com/google-research/google-research/tree/master/mathwriting),
[Pix2tex](https://github.com/lukas-blecher/LaTeX-OCR), and
[Texify](https://github.com/VikParuchuri/texify).  The bundled CDM tokenizer
includes third-party components from
[im2markup](https://github.com/harvardnlp/im2markup),
[KaTeX](https://github.com/KaTeX/KaTeX), and
[match-at](https://github.com/galkn/match-at).
