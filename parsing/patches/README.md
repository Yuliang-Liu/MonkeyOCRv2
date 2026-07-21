# Reproducible DFlash vLLM patch

`vllm-dflash.patch` is a source patch for the vLLM DFlash integration used by
the MonkeyOCRv2 serving path. It is intended for a clean checkout of the
official vLLM repository at base commit
`dbc3d9991ab0e5adc0db6a8c71c9059268032a14`.

The DFlash source used to produce this patch was an uncommitted local source
snapshot without Git metadata. Therefore this repository does not claim a
DFlash commit hash. The patch is a reviewable source diff from the official
base to that snapshot, limited to runtime/speculative-decoding integration
files and model registration files needed by the serving path.

## Apply and verify

```bash
git clone https://github.com/vllm-project/vllm.git <clean-vllm>
git -C <clean-vllm> checkout dbc3d9991ab0e5adc0db6a8c71c9059268032a14
git -C <clean-vllm> apply --check --whitespace=error \
  <path-to-repo>/parsing/patches/vllm-dflash.patch
git -C <clean-vllm> apply --whitespace=error \
  <path-to-repo>/parsing/patches/vllm-dflash.patch
```

After applying, the source should provide the DFlash speculative config and
proposer, the vLLM v1 DFlash worker path, the required model registrations,
and the backend adapters used by the launch scripts. Run the repository's
environment and real-image validation scripts after installation.

The patch contains source text only. It excludes model weights, images,
logs, Python bytecode, cache directories, build outputs, and compiled
extensions. It does not contain credentials or machine-specific paths.
