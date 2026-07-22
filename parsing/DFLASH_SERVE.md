# MonkeyOCRv2 DFlash vLLM

This directory provides a FlashAttention-backed DFlash speculative decoding
path for MonkeyOCRv2. Without `--draft-model`, `serve.py` keeps ordinary vLLM
serving behavior.

## 1. Install the vLLM patch

Use a clean local vLLM checkout at the commit expected by the patch. The
installer is offline and builds the native FlashAttention extension in the
same environment as vLLM.

```bash
bash parsing/scripts/install_dflash_vllm.sh \
  --vllm-source <clean-vllm-checkout> \
  --patch parsing/patches/vllm-dflash.patch \
  --flash-attn-source <local-vllm-flash-attn-source> \
  --cuda-home <cuda-toolkit> \
  --python <python>
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
