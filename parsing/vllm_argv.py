"""Assemble the ``vllm serve`` argv without importing vLLM."""
from __future__ import annotations

import json
from pathlib import Path


def build_vllm_argv(args) -> list[str]:
    """Build the vLLM command and add DFlash only when a draft is supplied."""
    argv = [
        "vllm",
        "serve",
        args.model_path,
        "--tensor-parallel-size",
        str(args.tensor_parallel_size),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--max-model-len",
        str(args.max_model_len),
        "--max-num-batched-tokens",
        str(args.max_num_batched_tokens),
        "--served-model-name",
        args.served_model_name,
        "--port",
        str(args.port),
    ]
    if args.max_num_seqs:
        argv.extend(["--max-num-seqs", str(args.max_num_seqs)])
    if args.host:
        argv.extend(["--host", args.host])

    if args.target_attention_backend:
        argv.extend(["--attention-backend", args.target_attention_backend])

    if args.draft_model:
        draft_model = str(Path(args.draft_model).expanduser())
        speculative_config = {
            "method": "dflash",
            "model": draft_model,
            "num_speculative_tokens": args.dflash_num_speculative_tokens,
        }
        if args.dflash_attention_backend:
            speculative_config["attention_backend"] = args.dflash_attention_backend
        argv.extend(["--speculative-config", json.dumps(speculative_config)])

    argv.append("--trust-remote-code")
    return argv
