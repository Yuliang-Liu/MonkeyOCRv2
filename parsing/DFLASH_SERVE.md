# MonkeyOCRv2 DFlash vLLM

MonkeyOCRv2 uses native DFlash speculative decoding from `vllm==0.25.1`.
No vLLM source patch, separate FlashAttention build, or compiled binary is
included or required. Without `--draft-model`, `serve.py` keeps ordinary vLLM
serving behavior.

## 1. Install vLLM

Install vLLM without model arguments. The model directories are checked only
after the DFlash model is downloaded:

```bash
bash parsing/scripts/install_dflash_vllm_pip.sh \
  --python python
```

If `vllm==0.25.1` is already installed, skip installation and continue with
model download.

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

## 3. Check the environment and models

```bash
python - <<'PY'
import vllm
print(vllm.__version__)
PY

python parsing/scripts/check_dflash_env.py \
  --require-native-dflash \
  --target-model ./models/MonkeyOCRv2-B-Parsing \
  --draft-model ./models/MonkeyOCRv2-B-Parsing-DFlash
```

The supported native route requires version `0.25.1`, `method=dflash`, the
native DFlash proposer, a draft architecture that resolves to
`DFlashMonkeyOCRv2ForCausalLM`, and vLLM's bundled `FLASH_ATTN` backend. The
check fails instead of silently falling back to ordinary decoding.

## 4. Start and call the service

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

The optional DFlash arguments are documented in `python parsing/serve.py
--help`: the default proposal block is 16 tokens and the DFlash-only default
for `--dflash-max-num-seqs` is 128. In DFlash mode, `serve.py` may adjust
`--max-num-batched-tokens` up to 65536, but never above
`--dflash-max-num-seqs * --max-model-len`; the resolved value is printed.
Override the sequence limit only after verifying GPU memory headroom.

```bash
python parsing/parse.py \
  --input-path ./images_test \
  --output-path ./output/dflash \
  --model-path ./models/MonkeyOCRv2-B-Parsing \
  --server-url http://127.0.0.1:8888 \
  --served-model-name MonkeyOCRv2
```
