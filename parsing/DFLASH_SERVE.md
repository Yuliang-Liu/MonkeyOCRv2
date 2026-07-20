# MonkeyOCRv2 DFlash vLLM Serve

The parsing API is already an OpenAI-compatible vLLM client. DFlash is enabled
only when the model server is started with a draft model; no changes are needed
in `core_runner.py` or `fastapi/main.py`.

## 1. Install the matching vLLM build

The stock vLLM package does not include the MonkeyOCRv2 DFlash model adapter.
Use a DFlash-enabled vLLM build that is API-compatible with the current
MonkeyOCRv2 model plugin. The build needs the DFlash scheduler/speculator changes
and the Qwen2.5-VL DFlash adapter, then must accept `--speculative-config` with
`"method": "dflash"`. Verify it before serving:

```bash
python -c "from vllm.config import SpeculativeConfig; print('DFlash vLLM import OK')"
python -m vllm.entrypoints.cli.main serve --help | grep speculative-config
```

## 2. Download the draft

After publishing the model, use one of the following commands. The target model
is still `MonkeyOCRv2-B-Parsing`; only the draft comes from the new repository.

```bash
python parsing/scripts/download_dflash_model.py \
  --source hf \
  --repo-id <org>/MonkeyOCRv2-B-Parsing-DFlash \
  --output-dir ./models/MonkeyOCRv2-B-Parsing-DFlash

python parsing/scripts/download_dflash_model.py \
  --source modelscope \
  --repo-id <namespace>/MonkeyOCRv2-B-Parsing-DFlash \
  --output-dir ./models/MonkeyOCRv2-B-Parsing-DFlash
```

## 3. Start the DFlash server

`--draft-model` is optional. Without it, `serve.py` starts the unchanged
baseline server. With it, the launcher adds the DFlash speculative configuration,
uses a 16-token proposal block, sets `max_num_seqs=128` by default and raises scheduler capacity to at least
65536. `128` is validated on a 24GB RTX 4090; use `--dflash-max-num-seqs 1024`
only on a GPU with sufficient warmup headroom.

```bash
cd parsing
python serve.py \
  --model-path ../models/MonkeyOCRv2-B-Parsing \
  --draft-model ./models/MonkeyOCRv2-B-Parsing-DFlash \
  --served-model-name MonkeyOCRv2 \
  --port 8888
```

The parsing API continues to call the same endpoint:

```bash
python fastapi/main.py \
  --model-path ../models/MonkeyOCRv2-B-Parsing \
  --server-url http://127.0.0.1:8888 \
  --served-model-name MonkeyOCRv2
```

## Compatibility and fallback

- Baseline remains the default: no draft path means no speculative decoding.
- Keep target and draft tokenizer, vocab size, image processor, and Qwen2.5-VL
  architecture compatible. Do not use the DFlash checkpoint as the target model.
- Do not reuse the legacy DFlash vLLM `0.11.2.dev0` environment directly. It
  lacks the `vllm.attention` API used by current MonkeyOCRv2 model code. Rebase
  its DFlash changes onto the official `vLLM==0.11.2` dependency from this repo
  before packaging the DFlash environment.
- `FLASHINFER` is passed only on the DFlash route because the validated DFlash
  environment uses it. Set `--dflash-attention-backend ''` if the deployed
  vLLM build requires its own default backend.
- A failed draft download or unsupported DFlash vLLM build stops startup before
  serving. It does not silently switch a requested DFlash service to baseline.

## Validated smoke configuration

The integration was exercised on a 24GB RTX 4090 with the target and a local
b16 draft. The real server loaded `DFlashMonkeyOCRv2ForCausalLM`, reported
`SpeculativeConfig(method='dflash')`, and completed an image chat-completion
request. The tested launch configuration was:

```bash
CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=<dflash-vllm-src>:<MonkeyOCRv2>/parsing \
PATH=<venv>/bin:$PATH \
python parsing/serve.py \
  --model-path ../models/MonkeyOCRv2-B-Parsing \
  --draft-model ./models/MonkeyOCRv2-B-Parsing-DFlash \
  --dflash-max-num-seqs 128 \
  --max-num-batched-tokens 65536
```

Before publishing, retain a `config.json` whose `dflash_config.block_size` is
`16` and whose architecture resolves to `DFlashMonkeyOCRv2ForCausalLM` in the
DFlash vLLM registry.
