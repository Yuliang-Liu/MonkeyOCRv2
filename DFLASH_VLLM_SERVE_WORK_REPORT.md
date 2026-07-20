# MonkeyOCRv2 DFlash vLLM Serve Work Report

## Scope

This change adds a DFlash route to the current `parsing/serve.py` vLLM-server
workflow. It does not modify target or draft weights, training code, parsing
prompts, or the OpenAI-compatible client contract.

## Delivered changes

- `parsing/serve.py`
  - Adds optional `--draft-model`.
  - Builds `--speculative-config` with `method=dflash`, the draft path, and
    `num_speculative_tokens=16`.
  - Enables FlashInfer only on the DFlash route and raises scheduler token
    capacity to `65536` when necessary.
  - Defaults DFlash `max_num_seqs` to `128`; it can be overridden explicitly.
- `parsing/scripts/download_dflash_model.py`
  - Downloads the future `MonkeyOCRv2-B-Parsing-DFlash` repository from
    Hugging Face or ModelScope into a supplied local directory.
- `parsing/DFLASH_SERVE.md`
  - Documents environment, download, server startup, parser invocation, and
    24GB GPU sizing.
- `parsing/modeling/modeling_monkeyocrv2_vllm.py`
  - Ports the required visual-attention imports to the DFlash fork's v1 API.
  - Uses a correct Torch SDPA visual-attention fallback where that fork no
    longer exports legacy `XFORMERS` helpers.
  - Declares the outer MonkeyOCRv2 target as `SupportsEagle3`; its existing
    inner Qwen3 language model supplies the auxiliary hidden-state interface
    required by DFlash verification.

## Environment validated

- DFlash-enabled vLLM source tree with DFlash scheduler/speculator and
  `DFlashMonkeyOCRv2ForCausalLM` registry adapter.
- PyTorch 2.11.0 + CUDA 12.8, Transformers 5.13.1, FlashInfer, and `ninja` on
  `PATH` for FlashInfer JIT.
- Target: local MonkeyOCRv2-B-Parsing checkpoint.
- Draft: local b16 DFlash checkpoint with `block_size=16` and five draft
  layers.
- GPU: one 24GB RTX 4090.

## Runtime validation

1. `py_compile` passed for the launcher, downloader, and model plugin.
2. Download script CLI help passed.
3. A real DFlash `vllm serve` process loaded both target and draft, reported:
   - `SpeculativeConfig(method='dflash')`
   - `DFlashMonkeyOCRv2ForCausalLM`
   - FlashInfer backend
   - DFlash speculator capture completed.
4. `GET /v1/models` returned the configured served model.
5. A vision `POST /v1/chat/completions` request completed successfully.
6. Current `parse.py --server-url` completed a one-image text recognition
   smoke in 1.89 seconds and wrote both Markdown and JSON outputs.

## Operational notes

- On the validated 24GB GPU, `max_num_seqs=1024` fails during speculative
  sampler warmup because it requests an additional approximately 9.85 GiB
  FP32 logits buffer. `128` is the safe default here.
- `1024` remains exposed as an explicit option for a larger GPU after a local
  memory check.
- FlashInfer may JIT kernels on first startup; `ninja` must be discoverable in
  the service process `PATH`.
- The visual SDPA fallback is correct and exercised by smoke, but benchmark it
  before treating this compatibility fork as a throughput-optimized release.
- No performance benchmark or production PR was created in this task.
