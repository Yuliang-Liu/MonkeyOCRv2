#!/usr/bin/env python3
import argparse
import json
import os
import socket
import sys
import inspect
from pathlib import Path

from modeling import modeling_monkeyocrv2_vllm  # noqa: F401
from vllm.entrypoints.cli.main import main as vllm_main


PARSING_DIR = Path(__file__).resolve().parent


def ensure_model_path(model_path: str):
    path = Path(model_path).expanduser()
    if not path.exists():
        raise SystemExit(f"Model path does not exist: {path}")


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
        ensure_model_path(draft_model)
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


def ensure_port_available(host: str | None, port: int):
    bind_host = host or "0.0.0.0"
    probe_host = "127.0.0.1" if bind_host in {"0.0.0.0", "::"} else bind_host
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(1.0)
        if sock.connect_ex((probe_host, port)) == 0:
            raise SystemExit(f"Port is already in use: {probe_host}:{port}")


def ensure_dflash_support() -> None:
    try:
        from vllm.config import SpeculativeConfig

        source = inspect.getsource(SpeculativeConfig).lower()
    except Exception as exc:
        raise SystemExit(f"DFlash requested but vLLM speculative config is unavailable: {exc}") from exc
    if "dflash" not in source:
        raise SystemExit("DFlash requested but this vLLM build does not support method=dflash")
    try:
        from vllm.v1.spec_decode.dflash import DFlashProposer  # noqa: F401
    except Exception as exc:
        raise SystemExit(
            "DFlash requested but vLLM has no usable native DFlash proposer: "
            f"{exc}"
        ) from exc


def main():
    parser = argparse.ArgumentParser(description="Start vLLM serve for MonkeyOCRv2.")
    parser.add_argument("--model-path", "-m", default=PARSING_DIR.parent / 'model_weight' / 'MonkeyOCRv2-B-Parsing')
    parser.add_argument("--tensor-parallel-size", "--tp", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    parser.add_argument("--max-model-len", type=int, default=16384)
    parser.add_argument("--max-num-seqs", type=int, default=128)
    parser.add_argument(
        "--max-num-batched-tokens",
        type=int,
        default=16384,
        help=(
            "Scheduler token budget; including speculative tokens."
        ),
    )
    parser.add_argument(
        "--target-attention-backend",
        type=str,
        help="Target attention backend.",
    )
    parser.add_argument(
        "--draft-model", "-d",
        type=str,
        help="Optional local path to a MonkeyOCRv2 DFlash draft. Enables DFlash speculative decoding.",
    )
    parser.add_argument(
        "--dflash-num-speculative-tokens",
        type=int,
        default=16,
        help="DFlash proposal block size. MonkeyOCRv2-B-Parsing-DFlash b16 uses 16.",
    )
    parser.add_argument(
        "--dflash-attention-backend",
        type=str,
        help="DFlash attention backend.",
    )
    parser.add_argument("--served-model-name", default="MonkeyOCRv2")
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", "-p", type=int, default=8888)
    parser.add_argument("extra_args", nargs=argparse.REMAINDER, help="Extra arguments passed to vLLM serve")
    args = parser.parse_args()

    ensure_model_path(args.model_path)
    if args.draft_model:
        ensure_model_path(args.draft_model)
        ensure_dflash_support()
        if args.dflash_num_speculative_tokens <= 0:
            parser.error("--dflash-num-speculative-tokens must be positive in DFlash mode")

    ensure_port_available(args.host, args.port)

    parsing_dir = Path(__file__).resolve().parent
    os.environ["PYTHONPATH"] = str(parsing_dir) + os.pathsep + os.environ.get("PYTHONPATH", "")

    argv = build_vllm_argv(args)
    if args.extra_args:
        if args.extra_args[0] == "--":
            args.extra_args = args.extra_args[1:]
        argv.extend(args.extra_args)

    print("Imported modeling.modeling_monkeyocrv2_vllm for vLLM model registration.")
    print("Running:", " ".join(argv))
    sys.argv = argv
    vllm_main()


if __name__ == "__main__":
    main()
