# MonkeyOCRv2 DFlash Backend Validation

This file records runtime validation only. The smoke timings are startup plus
one image request; they are not throughput or performance benchmarks.

## Backend smoke results

| Mode | Target backend | Draft backend | DFlash config | Real image | Result |
|---|---|---|---|---|---|
| DFlash + FlashAttention | `FLASH_ATTN` | `FLASH_ATTN` | `method=dflash`, 16 speculative tokens | PASS | PASS |
| DFlash + FlashInfer | `FLASHINFER` | `FLASHINFER` | `method=dflash`, 16 speculative tokens | PASS | PASS |

Measured smoke elapsed times were `93.01 s` for FlashAttention and `60.03 s`
for FlashInfer. These values include engine startup and must not be interpreted
as a backend speed comparison.

The validator required all of the following before reporting PASS:

- the vLLM server became ready;
- a real image chat-completion returned non-empty content;
- the server log contained the DFlash speculative configuration; and
- the server log contained the requested backend marker.

The FlashAttention compatibility environment also exercised the native FA2
CUDA call. The compatibility smoke used an existing matching local extension
and an opt-in ABI adapter; no compiled binary is part of this repository.

## Formal FlashAttention deployment

Do not copy `_vllm_fa2_C` or any other compiled extension between PyTorch/CUDA
environments. Use `scripts/build_vllm_fa2.sh` with a clean local vLLM worktree,
a local vllm-flash-attn source tree, and the target Python/PyTorch/CUDA
environment. The script builds the extension in that environment, then checks
that the FA2 Python extension imports. A real image validation with
`scripts/validate_dflash_backend.py` is still required after the build.

## Model architecture note

MonkeyOCRv2 uses a Qwen3 language backbone and Qwen2-VL-compatible vision
processing/attention components. The DFlash model is a separate text draft
model; it does not replace the target checkpoint or the vision encoder.

## Reproducible entry points

- `scripts/build_vllm_fa2.sh`: offline, same-environment FA2 build.
- `scripts/run_dflash_flashattn.sh`: explicit target and draft
  `FLASH_ATTN` launch with DFlash defaults.
- `scripts/check_dflash_env.py`: static imports and extension diagnostics.
- `scripts/validate_dflash_backend.py`: one real-image runtime validation.
