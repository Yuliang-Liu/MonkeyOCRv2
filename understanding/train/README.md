# Fine-Tuning MonkeyOCRv2-Und

## 1. Installation

Use the same environment with [`parsing/train`](../../parsing/train/README.md#1-installation):

```bash
conda create -n monkeyocrv2-train python=3.11 -y
conda activate monkeyocrv2-train

# Example for CUDA 12.6. Adjust this command for your CUDA environment.
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

pip install transformers==4.57.1 accelerate==1.11.0 qwen_vl_utils==0.0.14
pip install flash-attn==2.8.3 --no-build-isolation
pip install -e ../../parsing/train/ms-swift
```

Run the commands above from this `train` directory. `flash-attn` is recommended to reduce memory usage and accelerate training, but it can be omitted if it is not supported by your hardware.

## 2. Prepare the Dataset

Prepare the training data as a JSONL file. Each line should contain one sample in the ms-swift multimodal conversation format:

```json
{"messages": [{"role": "user", "content": "<image>QUESTION"}, {"role": "assistant", "content": "ANSWER"}], "images": ["/absolute/path/to/image.jpg"]}
```

## 3. Prepare model

Download MonkeyOCRv2-Und from HuggingFace:
```bash
python ../../download_model.py -n MonkeyOCRv2-B-Und # or MonkeyOCRv2-S-Und
```
You can also download MonkeyOCRv2-Und from ModelScope:
```bash
python ../../download_model.py -t modelscope -n MonkeyOCRv2-B-Und # or MonkeyOCRv2-S-Und
```

## 4. Training

The following examples use `MonkeyOCRv2-B-Und`. You may replace it with `MonkeyOCRv2-S-Und`. You should replace `/path/to/train.jsonl` with your own dataset path and adjust the batch size, sequence length, gradient accumulation, and number of GPUs according to your data and hardware. You can customize the training configuration by modifying the arguments in the training scripts. For a complete list of available options, please refer to the [official ms-swift documentation](https://swift.readthedocs.io/zh-cn/v3.11/Instruction/Command-line-parameters.html).

*Note: The original MonkeyOCRv2-Und is trained with max pixels of 1003520 and max sequence length of 8196.*

```bash
export CUDA_VISIBLE_DEVICES=0
export NPROC_PER_NODE=1
# Full-Parameter SFT
bash scripts/full.sh
# LoRA Fine-Tuning
bash scripts/lora.sh
```

In the training script, keeping `--freeze_vit true` preserves the visual representations learned by MonkeyOCRv2 and reduces GPU memory usage. For domains with a substantial visual shift, you can experiment with unfreezing the ViT, although this generally requires more memory, a smaller learning rate (configured by `--vit_lr`), and careful validation.