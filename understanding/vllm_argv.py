"""Build the vLLM serve argv for MonkeyOCRv2 understanding.

This module must not import vLLM so the command contract can be tested in CI.
"""


def build_vllm_argv(args) -> list[str]:
    argv = [
        "vllm",
        "serve",
        str(args.model_path),
        "--tensor-parallel-size",
        str(args.tensor_parallel_size),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--max-model-len",
        str(args.max_model_len),
        "--max-num-seqs",
        str(args.max_num_seqs),
        "--max-num-batched-tokens",
        str(args.max_num_batched_tokens),
        "--served-model-name",
        args.served_model_name,
        "--port",
        str(args.port),
        "--trust-remote-code",
    ]
    if args.host:
        argv += ["--host", args.host]
    return argv
