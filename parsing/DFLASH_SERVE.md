# MonkeyOCRv2 DFlash vLLM Serve

This directory adds an optional DFlash speculative-decoding route to the
MonkeyOCRv2 parsing server. The ordinary vLLM route remains the default when
no draft model is supplied.

## 1. Architecture and backend ownership

MonkeyOCRv2 uses a Qwen3 language backbone with Qwen2-VL-compatible vision
components. The target model is the normal MonkeyOCRv2 parsing checkpoint. A
DFlash checkpoint is a separate text draft model; it must not replace the
target model.

`parsing/serve.py` registers the MonkeyOCRv2 vLLM model plugin and constructs
the vLLM `SpeculativeConfig(method="dflash")` only when `--draft-model` is
present. The plugin supports the vLLM v1 model interface, `SupportsEagle3`, and
the Torch SDPA vision-attention fallback used when the selected vLLM build does
not expose the legacy vision-attention helpers.

The two attention settings have different owners:

| Setting | vLLM destination | Meaning |
|---|---|---|
| `--target-attention-backend` / `MOCR2_TARGET_ATTENTION_BACKEND` | global `--attention-backend` | optional target-model backend |
| `--dflash-attention-backend` / `MOCR2_DFLASH_ATTENTION_BACKEND` | `speculative_config.attention_backend` | DFlash draft backend only |

Supported explicit values are `FLASH_ATTN` and `FLASHINFER`. Empty values leave
that side on the vLLM default. A DFlash backend is never silently copied to the
global target backend.

## 2. Offline prerequisites

The server-side workflow is offline. It must use a vLLM source tree, model
directories, and dependency wheels that already exist locally. No script in
this change clones, fetches, pulls, pushes, or contacts a package index.

The selected vLLM environment must expose all of the following:

- vLLM v1 APIs used by `modeling/modeling_monkeyocrv2_vllm.py`.
- `vllm.v1.spec_decode.dflash.DFlashProposer` and a speculative config that
  accepts `method="dflash"`.
- The requested attention backend at runtime, not only as an enum entry.
- A compatible MonkeyOCRv2 target and DFlash draft checkpoint.

A stock vLLM wheel may contain the vLLM name and version but still lack the
DFlash scheduler/speculator. The environment check and real backend validator
must be used before claiming DFlash support.

For FlashInfer, `flashinfer-python` must be installed in the selected local
environment. FlashAttention is provided by the native FA2 extension built by
the selected vLLM source tree; this project does not install the unrelated
`flash-attn` Python package as a substitute. Missing optional dependencies are
reported as missing; they are not replaced by an unrequested backend.

## 3. Offline environment setup

Use an existing local vLLM worktree and, for FlashInfer, a local wheel
directory:

```bash
bash parsing/scripts/install_dflash_vllm.sh \
  --vllm-source <local-vllm-source> \
  --backend FLASHINFER \
  --flashinfer-wheel-dir <local-wheel-dir>
```

The installer checks the worktree, optionally applies a user-supplied local
patch after `git apply --check`, installs the local source with `--no-index`
and `--no-deps`, and verifies imports. It does not generate a vLLM patch. Do
not pass an unreviewed or dirty worktree: the installer refuses dirty source
trees to avoid overwriting existing work.

The optional dependency list is
`parsing/requirements-dflash-flashinfer.txt`. Install it only from an already
available local wheel/cache. FlashAttention does not require adding
`flash-attn` through this project, but it does require a native FA2 build.

Build FA2 in the target Python/PyTorch/CUDA environment with a local vLLM
worktree and local vllm-flash-attn source:

```bash
bash parsing/scripts/build_vllm_fa2.sh \
  --vllm-source <local-vllm-worktree> \
  --flash-attn-source <local-vllm-flash-attn-source> \
  --python <target-python> \
  --cuda-home <target-cuda-toolkit>
```

The build script is offline-only and compiles `_vllm_fa2_C` in place. It never
copies a compiled extension from another PyTorch/CUDA environment.

## 4. Download the draft model

The helper supports both registries. It downloads only when the user explicitly
runs it in an environment with the corresponding client and network access.

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

The draft must have the model configuration expected by the DFlash vLLM build,
including a proposal block size matching the launch setting. Keep target and
draft tokenizer/vocabulary and hidden-size compatibility aligned.

## 5. Start the server

Baseline vLLM mode is unchanged because it has no `--draft-model`:

```bash
cd parsing
python serve.py \
  --model-path ../models/MonkeyOCRv2-B-Parsing \
  --served-model-name MonkeyOCRv2 \
  --port 8888
```

DFlash with FlashInfer:

```bash
cd parsing
python serve.py \
  --model-path ../models/MonkeyOCRv2-B-Parsing \
  --draft-model ../models/MonkeyOCRv2-B-Parsing-DFlash \
  --dflash-attention-backend FLASHINFER \
  --dflash-max-num-seqs 128 \
  --served-model-name MonkeyOCRv2 \
  --port 8888
```

DFlash with FlashAttention uses the same launcher, changing only the draft
backend:

```bash
python serve.py \
  --model-path ../models/MonkeyOCRv2-B-Parsing \
  --draft-model ../models/MonkeyOCRv2-B-Parsing-DFlash \
  --target-attention-backend FLASH_ATTN \
  --dflash-attention-backend FLASH_ATTN \
  --dflash-max-num-seqs 128 \
  --served-model-name MonkeyOCRv2 \
  --port 8888
```

To explicitly select the target backend as well, add
`--target-attention-backend FLASHINFER` or `FLASH_ATTN`. This is independent
of the DFlash draft backend.

The launcher defaults to `num_speculative_tokens=16`. In DFlash mode it raises
`max_num_batched_tokens` to at least `65536` when a smaller value was supplied.
It does not add speculative arguments in baseline mode.

## 6. Call the parsing API

The parsing client continues to use the OpenAI-compatible vLLM endpoint:

```bash
python fastapi/main.py \
  --model-path ../models/MonkeyOCRv2-B-Parsing \
  --server-url http://127.0.0.1:8888 \
  --served-model-name MonkeyOCRv2
```

Use the repository's normal parsing/evaluation entry point for document
requests. The DFlash change is server-side recognition speculation; it does
not change layout preprocessing or the parsing API contract.

## 7. 24GB GPU guidance

The conservative DFlash setting is:

```text
gpu_memory_utilization=0.5
max_model_len=16384
max_num_batched_tokens=65536
dflash-max-num-seqs=128
num-speculative-tokens=16
```

`128` is the conservative starting point used by this project for a 24GB
deployment; it is not a universal hardware default. `1024` is not a universal
24GB default: use it only after a separate warmup and memory check on the
target GPU. Preprocessor and vLLM engine placement may need separate devices
when the preprocessor also holds large weights.

## 8. Checks and real validation

Static environment checks do not claim runtime support:

```bash
python parsing/scripts/check_dflash_env.py \
  --backend FLASHINFER \
  --model-path ../models/MonkeyOCRv2-B-Parsing \
  --draft-model ../models/MonkeyOCRv2-B-Parsing-DFlash
```

Run a real image validator only when the GPU and a compatible local model are
available:

```bash
python parsing/scripts/validate_dflash_backend.py \
  --backend FLASHINFER \
  --model-path ../models/MonkeyOCRv2-B-Parsing \
  --draft-model ../models/MonkeyOCRv2-B-Parsing-DFlash \
  --image <local-document-image> \
  --report <output-report> \
  --log-dir <output-log-dir>
```

Repeat with `--backend FLASH_ATTN` for the other route. A successful result
requires a non-empty image response plus server-log evidence of
`method=dflash`, `SpeculativeConfig`, and the requested backend. If any
prerequisite is missing, record `SKIPPED` and the concrete reason; do not turn
a static import or an enum lookup into a PASS.

## 9. Known limitations

- DFlash support depends on the selected vLLM source build; the ordinary vLLM
  wheel is not automatically upgraded or patched.
- FlashInfer requires a compatible local `flashinfer-python` build.
- FlashAttention availability is owned by the selected vLLM environment and
  must be verified at runtime.
- Target and draft backends can be selected independently, but their kernels
  still need to be compatible with the installed CUDA, GPU, and vLLM build.
