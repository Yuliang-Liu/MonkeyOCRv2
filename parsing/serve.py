#!/usr/bin/env python3
import argparse
import os
import socket
import sys
from pathlib import Path

from modeling import modeling_monkeyocrv2_vllm  # noqa: F401
from vllm.entrypoints.cli.main import main as vllm_main



def ensure_model_path(model_path: str):
    path = Path(model_path).expanduser()
    if not path.exists():
        raise SystemExit(f"Model path does not exist: {path}")


def ensure_port_available(host: str | None, port: int):
    bind_host = host or "0.0.0.0"
    probe_host = "127.0.0.1" if bind_host in {"0.0.0.0", "::"} else bind_host
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(1.0)
        if sock.connect_ex((probe_host, port)) == 0:
            raise SystemExit(f"Port is already in use: {probe_host}:{port}")


def main():
    parser = argparse.ArgumentParser(description="Start vLLM serve for MonkeyOCRv2.")
    parser.add_argument("--model-path", "-m", default='../model_weight/MonkeyOCRv2-B-Parsing')
    parser.add_argument("--tensor-parallel-size", "--tp", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    parser.add_argument("--max-model-len", "--max_model_len", type=int, default=16384)
    parser.add_argument("--max-num-batched-tokens", type=int, default=16384)
    parser.add_argument("--served-model-name", default="MonkeyOCRv2")
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", "-p", type=int, default=8888)
    parser.add_argument("--trust-remote-code", action="store_true", default=True)
    parser.add_argument("extra_args", nargs=argparse.REMAINDER, help="Extra arguments passed to vLLM serve")
    args = parser.parse_args()

    ensure_model_path(args.model_path)
    ensure_port_available(args.host, args.port)

    parsing_dir = Path(__file__).resolve().parent
    os.environ["PYTHONPATH"] = str(parsing_dir) + os.pathsep + os.environ.get("PYTHONPATH", "")

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
    if args.host:
        argv.extend(["--host", args.host])
    if args.trust_remote_code:
        argv.append("--trust-remote-code")
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
