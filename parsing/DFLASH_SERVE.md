# MonkeyOCRv2 DFlash vLLM

This directory provides a FlashAttention-backed DFlash speculative decoding
path for MonkeyOCRv2. Without `--draft-model`, `serve.py` keeps ordinary vLLM
serving behavior.

## 1. Install vLLM

### Option A: native pip vLLM (preferred)

Recent vLLM releases include the DFlash proposer and FlashAttention backend.
This path does not apply the bundled patch or compile a local FlashAttention
extension:

```bash
bash parsing/scripts/install_dflash_vllm_pip.sh \
  --vllm-version 0.25.1 \
  --python python \
  --target-model /path/to/MonkeyOCRv2-B-Parsing \
  --draft-model /path/to/MonkeyOCRv2-B-Parsing-DFlash
```

The installer validates native `method=dflash`, the DFlash proposer,
`FLASH_ATTN`, the vLLM FlashAttention extension, and the MonkeyOCRv2 plugin.
It fails instead of silently falling back to ordinary decoding. vLLM 0.11.2
is too old for this native path; use the source-patch path below for that
version.

### Option B: bundled source patch (legacy fallback)

The bundled patch targets vLLM commit
`dbc3d9991ab0e5adc0db6a8c71c9059268032a14`. Keep the vLLM checkout and the
MonkeyOCRv2 checkout in separate directories:

```bash
git clone https://github.com/vllm-project/vllm.git /path/to/vllm
git -C /path/to/vllm checkout dbc3d9991ab0e5adc0db6a8c71c9059268032a14
cd /path/to/MonkeyOCRv2
```

The installer checks that this worktree is clean before applying the patch,
then builds the native FlashAttention extension in the same environment as
vLLM. The FA2 build explicitly allows the changes produced by this bundled
patch.

```bash
bash parsing/scripts/install_dflash_vllm.sh \
  --vllm-source /path/to/vllm \
  --patch parsing/patches/vllm-dflash.patch \
  --flash-attn-source /path/to/vllm-flash-attn \
  --cuda-home /usr/local/cuda \
  --python python
```

Use `--skip-fa2-build` only when the same environment already has a validated
FlashAttention FA2 extension. Do not copy compiled extensions between
environments.

## 2. Download the DFlash model

The helper supports Hugging Face and ModelScope. Replace the repository ID
with the published `MonkeyOCRv2-B-Parsing-DFlash` model.

```bash
python parsing/scripts/download_dflash_model.py \
  --source hf \
  --repo-id <org>/MonkeyOCRv2-B-Parsing-DFlash \
  --output-dir ./models/MonkeyOCRv2-B-Parsing-DFlash
```

```bash
python parsing/scripts/download_dflash_model.py \
  --source modelscope \
  --repo-id <org>/MonkeyOCRv2-B-Parsing-DFlash \
  --output-dir ./models/MonkeyOCRv2-B-Parsing-DFlash
```

## 3. Start and call the service

Ordinary vLLM serving:

```bash
python parsing/serve.py \
  --model-path ./models/MonkeyOCRv2-B-Parsing \
  --target-attention-backend FLASH_ATTN \
  --port 8888
```

DFlash speculative serving:

```bash
python parsing/serve.py \
  --model-path ./models/MonkeyOCRv2-B-Parsing \
  --draft-model ./models/MonkeyOCRv2-B-Parsing-DFlash \
  --target-attention-backend FLASH_ATTN \
  --dflash-attention-backend FLASH_ATTN \
  --validate-models \
  --port 8888
```

Use the existing `parsing/parse.py` client or the OpenAI-compatible endpoint
at `http://127.0.0.1:8888/v1/chat/completions`. The command-line help in
`parsing/serve.py` is the source of truth for the optional serving parameters;
the DFlash defaults are 16 speculative tokens and 128 maximum concurrent
sequences.

```bash
python parsing/parse.py \
  --input-path ./images_test \
  --output-path ./output/dflash \
  --model-path ./models/MonkeyOCRv2-B-Parsing \
  --server-url http://127.0.0.1:8888 \
  --served-model-name MonkeyOCRv2
```
