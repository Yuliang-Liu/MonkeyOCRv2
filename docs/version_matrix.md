# Version Compatibility Matrix

Subprojects in this repository use **different** Python, PyTorch, Transformers, and vLLM pins. The table below only restates combinations already written in the install guides. It does not introduce or endorse mixing pins across rows.

| Subproject | Python | PyTorch | Transformers | vLLM | Hardware |
|---|---|---|---|---|---|
| Vision Encoder | 3.11 | 2.8.0 (`cu126`) | 4.57.1 | — | NVIDIA GPU, CUDA 12.6 wheels |
| Document Understanding | 3.11 | same as Vision Encoder | 4.57.1 | — | NVIDIA GPU (same as Vision Encoder) |
| Document Parsing (GPU) | 3.11 | installed with vLLM (not pinned separately) | — | **0.25.1** with DFlash (CUDA 12.9+); **0.11.2** without DFlash | NVIDIA GPU |
| Document Parsing (CPU) | 3.11 | 2.5.1+cpu | 4.57.1 | not used | CPU |
| Scene Text Recognition | 3.10 | 2.10.0 (`cu128`) | 4.57.1 | — | NVIDIA GPU, CUDA 12.8 (reproduced on H800) |
| Formula Recognition | 3.10 | 2.x (`cu121` wheels; `torch>=2.2.2`) | `>=4.42.4` | — | NVIDIA GPU (4 GPUs for the documented training recipe) |
| Scene Text Detection | 3.11 | 2.9.0 (`cu128`) | 4.57.1 | — | NVIDIA GPU, CUDA 12.8, Detectron2 0.6 (reproduced on 8× RTX 3090) |

`—` means that package is not pinned, or not used, in that subproject's install instructions.

Rows that say "reproduced" are the environments used for the published results. Nearby versions are not claimed to work or fail.

## Notes

- **DFlash** (`MonkeyOCRv2-B-Parsing-DFlash`) is a vLLM speculative-decoding draft for GPU serving of MonkeyOCRv2-B-Parsing only. The CPU parsing path does not load it. See [Document Parsing](../README.md#document-parsing) and [CPU support](cpu_support.md).
- A WSL2 + NVIDIA GPU setup was tested with Python 3.11.16, PyTorch 2.11.0+cu129, and vLLM 0.25.1. See [WSL2 support](wsl_support.md). That is a tested environment for Document Parsing (GPU), not a separate subproject pin.
- Parsing GPU does not pin PyTorch or Transformers in the README install; those come from the vLLM wheel.
- Formula does not pin an exact PyTorch version. The README install uses the `cu121` index, and `formula/requirements.txt` requires `torch>=2.2.2`.

## Sources

| Subproject | Source |
|---|---|
| Vision Encoder | [README.md](../README.md#vision-encoder) |
| Document Understanding | [README.md](../README.md#document-understanding) ("See install part of MonkeyOCRv2 vision encoder") |
| Document Parsing (GPU) | [README.md](../README.md#document-parsing) |
| Document Parsing (CPU) | [cpu_support.md](cpu_support.md) |
| Recognition | [recognition/README.md](../recognition/README.md), [recognition/requirements.txt](../recognition/requirements.txt) |
| Formula | [formula/README.md](../formula/README.md), [formula/requirements.txt](../formula/requirements.txt) |
| Detection | [detection/DPText-DETR/README.md](../detection/DPText-DETR/README.md) |
| DFlash | [README.md](../README.md#document-parsing); `parsing/serve.py --draft-model` |
