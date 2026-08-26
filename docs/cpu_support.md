# MonkeyOCRv2-Parsing on CPU 

This guide covers running MonkeyOCRv2 Document Parsing on **CPU**. For GPU inference, see [Document Parsing](../README.md#document-parsing) in this repository.

CPU inference loads only the parsing checkpoint passed to `-m` / `--model-path` (`MonkeyOCRv2-B-Parsing` or `MonkeyOCRv2-S-Parsing`). Do **not** download `MonkeyOCRv2-B-Parsing-DFlash` for this guide: that draft model is used only for GPU vLLM speculative decoding.

---

## Setup MonkeyOCRv2-Parsing(Windows)

### 1. Clone the repository

```bash
git clone https://github.com/Yuliang-Liu/MonkeyOCRv2.git
cd MonkeyOCRv2
```

### 2. Create virtual environment

```bash
py -3.11 -m venv .venv
```

### 3. Install dependencies

```bash
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch==2.5.1+cpu torchvision==0.20.1+cpu torchaudio==2.5.1+cpu --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r parsing\requirements.txt
.\.venv\Scripts\python.exe -m pip install transformers==4.57.1 accelerate==1.11.0 huggingface_hub qwen_vl_utils opencv-python einops
```

### 4. Download model weights

From ModelScope (faster in China):

```bash
.\.venv\Scripts\python.exe -m pip install modelscope
python download_model.py -t modelscope -n MonkeyOCRv2-B-Parsing # or MonkeyOCRv2-S-Parsing
```

Or from HuggingFace:

```bash
python download_model.py -n MonkeyOCRv2-B-Parsing # or MonkeyOCRv2-S-Parsing
```

---

### Parse using CLI

```bash
cd .\parsing\
# Parse a single document
..\.venv\Scripts\python.exe cpu\parse_cpu.py `
    -i ..\images_test\vqa.png `
    -m ..\model_weight\MonkeyOCRv2-B-Parsing `
    -o output\test_cpu
# Parse a directory containing PDFs or images
..\.venv\Scripts\python.exe cpu\parse_cpu.py `
    -i ..\images_test `
    -m ..\model_weight\MonkeyOCRv2-B-Parsing `
    -o output\test_cpu
```

---

### Serve with Web Demo

```bash
cd .\parsing\
..\.venv\Scripts\python.exe cpu\gradio_demo_cpu.py `
    --model-path ..\model_weight\MonkeyOCRv2-B-Parsing `
    --output-dir output\demo_cpu_outputs `
    --demo-server-name 127.0.0.1 `
    --demo-server-port 8891 `
    --demo-concurrency 1 `
    --page-max-inflight 1
```

---


## Setup MonkeyOCRv2-Parsing(Linux)

### 1. Clone the repository

```bash
git clone https://github.com/Yuliang-Liu/MonkeyOCRv2.git
cd MonkeyOCRv2
```

### 2. Create virtual environment

```bash
conda create -n MonkeyOCRv2Parsing_CPU python=3.11 -y
conda activate MonkeyOCRv2Parsing_CPU
```

### 3. Install dependencies

```bash
pip install torch==2.5.1+cpu torchvision==0.20.1+cpu torchaudio==2.5.1+cpu --index-url https://download.pytorch.org/whl/cpu
pip install -r parsing/requirements.txt
pip install transformers==4.57.1 accelerate==1.11.0 huggingface_hub qwen_vl_utils opencv-python einops
```

### 4. Download model weights

From ModelScope (faster in China):

```bash
pip install modelscope
python download_model.py -t modelscope -n MonkeyOCRv2-B-Parsing # or MonkeyOCRv2-S-Parsing
```

Or from HuggingFace:

```bash
python download_model.py -n MonkeyOCRv2-B-Parsing # or MonkeyOCRv2-S-Parsing
```

---

### Parse using CLI

```bash
cd parsing
# Parse a single document
python cpu/parse_cpu.py \
    -i ../images_test/vqa.png \
    -m ../model_weight/MonkeyOCRv2-B-Parsing \
    -o output/test_cpu
# Parse a directory containing PDFs or images
python cpu/parse_cpu.py \
    -i ../images_test \
    -m ../model_weight/MonkeyOCRv2-B-Parsing \
    -o output/test_cpu
```

---

### Serve with Web Demo

```bash
cd parsing
python cpu/gradio_demo_cpu.py \
    --model-path ../model_weight/MonkeyOCRv2-B-Parsing \
    --output-dir output/demo_cpu_outputs \
    --demo-server-name 127.0.0.1 \
    --demo-server-port 8891 \
    --demo-concurrency 1 \
    --page-max-inflight 1
# Show help messages
python cpu/gradio_demo_cpu.py -h
```
