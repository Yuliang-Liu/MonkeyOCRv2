# Fine-Tuning MonkeyOCRv2-Parsing

MonkeyOCRv2-Parsing is trained with a deliberately simple recipe: a standard supervised fine-tuning (SFT) stage with the vision transformer (ViT) frozen. It does not rely on complex post-training pipelines. This makes the released checkpoint a strong and flexible starting point for downstream adaptation.

We warmly welcome the community to fine-tune MonkeyOCRv2-Parsing on data from different domains, languages, and document types. If you obtain interesting results, encounter any problems, or would like to discuss training recipes, please feel free to [open an issue](https://github.com/Yuliang-Liu/MonkeyOCRv2/issues).

This directory contains a MonkeyOCRv2-compatible version of [ms-swift](https://github.com/modelscope/ms-swift), which is used for training.

## 1. Installation

We recommend creating a clean Conda environment. Install a PyTorch build that matches the CUDA version on your machine, and then install the remaining dependencies and the bundled ms-swift package:

```bash
conda create -n monkeyocrv2-train python=3.11 -y
conda activate monkeyocrv2-train

# Example for CUDA 12.6. Adjust this command for your CUDA environment.
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

pip install transformers==4.57.1 accelerate==1.11.0 qwen_vl_utils==0.0.14
pip install flash-attn==2.8.3 --no-build-isolation
pip install -e ./ms-swift
```

Run the commands above from this `train` directory. `flash-attn` is recommended to reduce memory usage and accelerate training, but it can be omitted if it is not supported by your hardware.

## 2. Prepare the Dataset

Prepare the training data as a JSONL file. Each line should contain one sample in the ms-swift multimodal conversation format:

```json
{"messages": [{"role": "user", "content": "<image><PROMPT>"}, {"role": "assistant", "content": "<CONTENT>"}], "images": ["/absolute/path/to/image.jpg"]}
```

The number of `<image>` placeholders in `messages` must match the number of entries in `images`. Absolute image paths are recommended. The assistant response should follow the output format required by your downstream task.

The `<PROMPT>` and `<CONTENT>` formats for different tasks are listed below:

* **Text Recognition**
  `<PROMPT>`: 'Please output the text content from the image.'
  `<CONTENT>`: Plain text without Markdown formatting.

* **Formula Recognition**
  `<PROMPT>`: 'Please write out the expression of the formula in the image using LaTeX format.'
  `<CONTENT>`: A LaTeX expression wrapped in `$`. Example: `$C_{0}$`.

* **Table Recognition (HTML)**
  `<PROMPT>`: 'This is the image of a table. Please output the table in HTML format.'
  `<CONTENT>`: An HTML table wrapped with `<table>` and `</table>`, with no spaces between structural tags. Example: `<table><tr><td>A</td><td>B</td></tr></table>`.

* **Table Recognition (OTSL)**
  `<PROMPT>`: 'Please extract the table from the image and represent it in OTSL format.'
  `<CONTENT>`: OTSL converted from the HTML format above using:
  ```bash
  python html2otsl.py -i html.jsonl -o otsl.jsonl
  ```
  *<b>Note: For table recognition, we recommend using the OTSL format, which is more token-efficient than HTML.</b>*

* **Layout Detection**
  `<PROMPT>`: `Please output the categories and coordinates of the document elements in reading order.`
  `<CONTENT>`: A list of dictionaries containing `bbox` and `label`. Example:
  ```python
  [{'bbox': [x1, y1, x2, y2], 'label': '<LABEL>'}, {'bbox': [x1, y1, x2, y2], 'label': '<LABEL>'}]
  ```

The `<LABEL>` values used in **MonkeyOCRv2-Parsing** include `Caption`, `Footnote`, `List-item`, `Page-footer`, `Page-header`, `Section-header`, `Text`, `Title`, `Formula`, `Table`, and `Picture`. Among these, all categories except `Formula`, `Table`, and `Picture` use the **Text Recognition** prompt.


## 3. Training

The following examples use `zenosai/MonkeyOCRv2-B-Parsing`. You may replace it with `zenosai/MonkeyOCRv2-S-Parsing` or a local checkpoint path. You should replace `/path/to/train.jsonl` with your own dataset path and adjust the batch size, sequence length, gradient accumulation, and number of GPUs according to your data and hardware. You can customize the training configuration by modifying the arguments in the training scripts. For a complete list of available options, please refer to the [official ms-swift documentation](https://swift.readthedocs.io/zh-cn/v3.11/Instruction/Command-line-parameters.html).

*Note: The original MonkeyOCRv2-Parsing is trained with max pixels of 1003520 and max sequence length of 16384.*

```bash
export CUDA_VISIBLE_DEVICES=0
export NPROC_PER_NODE=1
# Full-Parameter SFT
bash scripts/full.sh
# LoRA Fine-Tuning
bash scripts/lora.sh
```

In the training script, keeping `--freeze_vit true` preserves the visual representations learned by MonkeyOCRv2 and reduces GPU memory usage. For domains with a substantial visual shift, you can experiment with unfreezing the ViT, although this generally requires more memory, a smaller learning rate (configured by `--vit_lr`), and careful validation.