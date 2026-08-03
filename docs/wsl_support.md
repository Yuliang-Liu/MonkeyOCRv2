# MonkeyOCRv2 on WSL2 (Windows Subsystem for Linux)

This guide covers running MonkeyOCRv2 Document Parsing on **WSL2 with an NVIDIA GPU**. If you are looking for native Windows support (without WSL), please refer to [MonkeyOCR Windows Support](https://github.com/Yuliang-Liu/MonkeyOCR/blob/main/docs/windows_support.md).

## Tested Environment

| Component | Version / Detail |
|---|---|
| **OS** | Windows 11 Home China (Build 26200) + WSL2 (Ubuntu) |
| **GPU** | NVIDIA GeForce RTX 5070 Laptop GPU (8GB VRAM) |
| **NVIDIA Driver** | 582.05 (CUDA 13.0 driver-level support) |
| **CUDA Runtime** | **12.8** (required by vLLM 0.11.2) |
| **Python** | 3.10 |
| **PyTorch** | 2.9.0+cu128 |
| **vLLM** | 0.11.2 |
| **Model** | MonkeyOCRv2-S-Parsing (0.6B params, ~1.5GB weights) |

---

## Prerequisites

### 1. WSL2 with GPU passthrough

```powershell
# In Windows PowerShell (Admin)
wsl --install -d Ubuntu
wsl --update
```

Ensure your NVIDIA driver supports WSL2 GPU. The driver from [nvidia.com/drivers](https://www.nvidia.com/download/index.aspx) includes WSL2 CUDA support. Verify inside WSL:

```bash
nvidia-smi
```

You should see your GPU listed with driver version and CUDA version.

### 2. Install uv (Python package manager)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.local/bin/env
uv --version  # should show uv 0.11.x or later
```

---

## Setup MonkeyOCRv2-Parsing

### 1. Clone the repository

```bash
git clone https://github.com/Yuliang-Liu/MonkeyOCRv2.git
cd MonkeyOCRv2
```

### 2. Create virtual environment

```bash
uv venv --python 3.10
source .venv/bin/activate
```

### 3. Install PyTorch with CUDA 12.8

> **Critical:** vLLM 0.11.2 requires CUDA 12.8 runtime (`libcudart.so.12`). Using `--torch-backend=auto` may incorrectly select CUDA 13.0 on newer drivers, causing runtime errors. Always specify `cu128` explicitly.

With Tsinghua mirror (recommended for users in China):

```bash
uv pip install torch==2.9.0 torchvision==0.24.0 \
  --index-url https://download.pytorch.org/whl/cu128 \
  -i https://pypi.tuna.tsinghua.edu.cn/simple
```

Or without mirror:

```bash
uv pip install torch==2.9.0 torchvision==0.24.0 \
  --index-url https://download.pytorch.org/whl/cu128
```

Verify CUDA runtime:

```bash
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}')"
# Expected: PyTorch 2.9.0+cu128, CUDA 12.8
```

### 4. Install vLLM and dependencies

```bash
uv pip install vllm==0.11.2 requests \
  -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install -r parsing/requirements.txt
```

### 5. Download model weights

From ModelScope (faster in China):

```bash
pip install modelscope
python download_model.py -t modelscope -n MonkeyOCRv2-S-Parsing
```

Or from HuggingFace:

```bash
python download_model.py -n MonkeyOCRv2-S-Parsing
```

---

## Fix vLLM Compatibility

MonkeyOCRv2 uses custom model code that vLLM 0.11.2 may not recognize out of the box. Two fixes are needed:

### Fix 1: Add `AutoModel` mapping in `config.json`

Edit `model_weight/MonkeyOCRv2-S-Parsing/config.json` and ensure `auto_map` includes:

```json
{
  "auto_map": {
    "AutoConfig": "configuration_monkeyocrv2.MonkeyOCRv2Config",
    "AutoModelForCausalLM": "modeling_monkeyocrv2.MonkeyOCRv2ForCausalLM",
    "AutoModel": "modeling_monkeyocrv2.MonkeyOCRv2ForCausalLM"
  }
}
```

### Fix 2: Add `text_config` for multimodal detection

vLLM determines whether a model is multimodal by comparing `hf_config != hf_text_config`. Without a `text_config` attribute, vLLM uses `TransformersForCausalLM` (text-only), which fails because the model has a vision encoder.

Edit `model_weight/MonkeyOCRv2-S-Parsing/configuration_monkeyocrv2.py` and add `text_config` in `MonkeyOCRv2Config.__init__`:

```python
class MonkeyOCRv2Config(Qwen3Config):
    model_type = "monkeyocrv2"
    def __init__(self,
        image_token_id = 151655,
        video_token_id = 151656,
        vision_config: Optional[dict] = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.image_token_id = image_token_id
        self.video_token_id = video_token_id
        self.vision_config = MonkeyOCRv2VisionConfig(**(vision_config or {}))
        # Required by vLLM for multimodal model detection
        text_kwargs = {k: v for k, v in self.to_dict().items()
                       if k in Qwen3Config().to_dict()}
        self.text_config = Qwen3Config(**text_kwargs)
```

### Fix 3: Use custom vLLM model registration

The repository includes a vLLM-registered model class at `parsing/modeling/modeling_monkeyocrv2_vllm.py`. Start the server from the `parsing/` directory with `PYTHONPATH` set:

```bash
cd parsing
PYTHONPATH=/path/to/MonkeyOCRv2/parsing python serve.py \
  --model-path ../model_weight/MonkeyOCRv2-S-Parsing \
  --gpu-memory-utilization 0.55 \
  --max-model-len 8192 \
  --max-num-batched-tokens 4096 \
  --port 8888 \
  -- --enforce-eager
```

> **Note:** `--enforce-eager` is used because WSL2 Ubuntu may lack a C++ compiler for Triton JIT compilation. Install `build-essential` if you want full compilation performance:
> ```bash
> sudo apt update && sudo apt install -y build-essential
> ```

---

## Start vLLM Service

```bash
cd parsing
PYTHONPATH=$(pwd) nohup python serve.py \
  --model-path ../model_weight/MonkeyOCRv2-S-Parsing \
  --gpu-memory-utilization 0.55 \
  --max-model-len 8192 \
  --max-num-batched-tokens 4096 \
  --port 8888 \
  -- --enforce-eager \
  > ../vllm_serve.log 2>&1 &
```

Wait for the service to be ready:

```bash
# Check if port is listening
ss -tlnp | grep 8888

# Or check health endpoint
curl -s http://127.0.0.1:8888/health
```

Check logs:

```bash
tail -f ../vllm_serve.log
# Look for: "Resolved architecture: MonkeyOCRv2ForCausalLM"
# And: "Starting vLLM API server 0 on http://0.0.0.0:8888"
```

Expected GPU memory usage after startup: **~4.5 GB** (for S-Parsing, 0.55 utilization, 8192 context).

---

## Test Inference

### CLI test (single image)

```bash
cd parsing
python parse.py \
  -i ../images_test \
  -o output/test \
  -s http://127.0.0.1:8888 \
  --skip-preprocess
```

### OpenAI-compatible API test

```python
import base64, io, requests
from PIL import Image

# Encode image
with Image.open("document.jpg") as img:
    img = img.convert("RGB")
    # Resize to ~1M pixels
    w, h = img.size
    max_px = 1003520
    if w * h > max_px:
        scale = (max_px / (w * h)) ** 0.5
        img = img.resize((max(1, int(w*scale)), max(1, int(h*scale))), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    data_uri = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

# Send request
resp = requests.post("http://127.0.0.1:8888/v1/chat/completions", json={
    "model": "MonkeyOCRv2",
    "temperature": 0,
    "max_tokens": 4096,
    "messages": [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": data_uri}},
        {"type": "text", "text": "Please output the text content from the image."}
    ]}]
})

print(resp.json()["choices"][0]["message"]["content"])
```

---

## Performance Benchmarks (RTX 5070 Laptop, eager mode)

Tested on 4 real-world document pages × 4 image pre-processing views (16 runs total), all with `max_tokens=4096`:

| Page | Original | CLAHE | Shadow Norm | Unsharp | Avg |
|---|---|---|---|---|---|
| Arabic newspaper (outdoor) | 210.3s | 66.0s | 62.7s | 61.9s | 100.2s |
| Arabic newspaper (indoor) | 144.2s | 67.4s | 62.2s | 63.7s | 84.4s |
| Japanese newspaper | 79.4s | 63.5s | 62.3s | 62.1s | 66.8s |
| Russian note | 46.1s | 45.8s | 48.8s | 46.0s | 46.7s |
| **Average** | **120.0s** | **60.7s** | **59.0s** | **58.4s** | |

> **Notes:**
> - All runs with `temperature=0`, `max_tokens=4096`, eager mode (no Triton JIT)
> - "Original" first-run times include cold-start overhead (~1-2 min for encoder warmup). Subsequent runs are ~45-65s.
> - With `build-essential` installed, compiled mode can reduce latency by ~15-25%.
> - GPU memory usage stable at ~4.5-4.8 GB throughout all runs.

### Quick benchmark command

```bash
cd parsing
PYTHONPATH=$(pwd) python -c "
import time, requests, base64, io
from PIL import Image

img = Image.open('../images_test/sample.png').convert('RGB')
w, h = img.size
scale = (1003520 / (w * h)) ** 0.5
img = img.resize((max(1, int(w*scale)), max(1, int(h*scale))), Image.Resampling.LANCZOS)
buf = io.BytesIO()
img.save(buf, format='PNG')
data_uri = 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode()

t0 = time.perf_counter()
r = requests.post('http://127.0.0.1:8888/v1/chat/completions', json={
    'model': 'MonkeyOCRv2', 'temperature': 0, 'max_tokens': 4096,
    'messages': [{'role': 'user', 'content': [
        {'type': 'image_url', 'image_url': {'url': data_uri}},
        {'type': 'text', 'text': 'Please output the text content from the image.'}
    ]}]
})
elapsed = time.perf_counter() - t0
tokens = r.json()['usage']['completion_tokens']
print(f'Elapsed: {elapsed:.1f}s, Tokens: {tokens}, Speed: {tokens/elapsed:.1f} tok/s')
"
```

---

## Common Issues

### 1. `ImportError: libcudart.so.12` or CUDA version mismatch

**Symptom:** vLLM fails to load with missing CUDA 12 shared libraries.

**Cause:** PyTorch was installed with CUDA 13.0 runtime, but vLLM 0.11.2 links against CUDA 12.8.

**Fix:** Reinstall PyTorch with explicit cu128:
```bash
uv pip install torch==2.9.0 --index-url https://download.pytorch.org/whl/cu128 --force-reinstall
```

### 2. `ValueError: There is no module or parameter named 'model.embed_tokens'`

**Symptom:** vLLM log shows `Resolved architecture: TransformersForCausalLM` (should be `TransformersMultiModalForCausalLM`).

**Cause:** vLLM doesn't detect the model as multimodal because `config.text_config` is missing.

**Fix:** Apply Fix 2 from the [Fix vLLM Compatibility](#fix-vllm-compatibility) section.

### 3. `ValueError: Unrecognized configuration class`

**Symptom:** vLLM fails with "Unrecognized configuration class" at startup.

**Cause:** `config.json` `auto_map` does not include `AutoModel`.

**Fix:** Apply Fix 1 from the [Fix vLLM Compatibility](#fix-vllm-compatibility) section.

### 4. Triton JIT compilation fails

**Symptom:** Service exits during first inference with compilation errors.

**Cause:** No C++ compiler installed in WSL2 Ubuntu.

**Fix:** Either:
```bash
sudo apt install -y build-essential  # enables full compilation
```
Or use `--enforce-eager` flag (slower but no compiler needed).

### 5. WSL2 network is very slow downloading packages

**Fix:** Use Tsinghua mirror:
```bash
export UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
```

### 6. `--torch-backend=auto` picks wrong CUDA version

On systems with CUDA 13.0 driver, `uv pip install vllm --torch-backend=auto` may select CUDA 13.0 PyTorch, which is incompatible with vLLM 0.11.2.

**Fix:** Always install PyTorch and vLLM separately, specifying cu128 explicitly.

---

## Stopping the Service

```bash
# Find the process
ps aux | grep "serve.py.*8888" | grep -v grep

# Graceful stop
kill <PID>

# Force stop if needed
kill -9 <PID>

# Verify GPU memory is freed
nvidia-smi
```

---

## File Structure After Setup

```
MonkeyOCRv2/
├── docs/
│   └── wsl_support.md          # This guide
├── model_weight/
│   └── MonkeyOCRv2-S-Parsing/
│       ├── config.json         # (modified: AutoModel mapping added)
│       ├── configuration_monkeyocrv2.py  # (modified: text_config added)
│       ├── model.safetensors   # ~1.5GB
│       └── ...
├── parsing/
│   ├── serve.py                # vLLM serve entry point
│   ├── modeling/
│   │   └── modeling_monkeyocrv2_vllm.py  # Custom vLLM model registration
│   └── ...
├── .venv/                      # Python virtual environment
└── vllm_serve.log              # Service log
```

---

## References

- [MonkeyOCRv2 Official Repository](https://github.com/Yuliang-Liu/MonkeyOCRv2)
- [MonkeyOCR Windows Support](https://github.com/Yuliang-Liu/MonkeyOCR/blob/main/docs/windows_support.md) — native Windows setup (non-WSL)
- [vLLM Installation Guide](https://docs.vllm.ai/en/v0.11.2/getting_started/installation/gpu/)
- [NVIDIA CUDA on WSL](https://docs.nvidia.com/cuda/wsl-user-guide/index.html)
