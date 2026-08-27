#!/usr/bin/env python3
"""Start the OpenAI-compatible vLLM server for MonkeyOCRv2 understanding."""
import argparse
import os
import re
import socket
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from vllm.entrypoints.cli.main import main as vllm_main

UNDERSTANDING_DIR = Path(__file__).resolve().parent
os.environ["PYTHONPATH"] = str(UNDERSTANDING_DIR) + os.pathsep + os.environ.get("PYTHONPATH", "")


def ensure_port_available(host: str | None, port: int) -> None:
    probe_host = "127.0.0.1" if not host or host in {"0.0.0.0", "::"} else host
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(1.0)
        if sock.connect_ex((probe_host, port)) == 0:
            raise SystemExit(f"Port is already in use: {probe_host}:{port}")


def vllm_version_tuple() -> tuple[int, ...]:
    try:
        raw = version("vllm")
    except PackageNotFoundError as exc:
        raise SystemExit("vLLM is not installed in the current Python environment.") from exc
    match = re.match(r"^(\d+(?:\.\d+)*)", raw)
    if not match:
        raise SystemExit(f"Unable to determine installed vLLM version: {raw}")
    return tuple(int(part) for part in match.group(1).split("."))


# vLLM worker processes may not execute main() from this script. Register the
# architecture while the launcher module is imported, as parsing/serve.py does.
_VLLM_VERSION = vllm_version_tuple()
if _VLLM_VERSION >= (0, 11) and _VLLM_VERSION < (0, 12):
    from modeling import modeling_monkeyocrv2und_vllm_011  # noqa: F401
elif _VLLM_VERSION >= (0, 25):
    from modeling import modeling_monkeyocrv2und_vllm  # noqa: F401
else:
    raise SystemExit(f"Unsupported vLLM version: {_VLLM_VERSION}. Requires vLLM >= 0.25 or == 0.11.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Start vLLM for MonkeyOCRv2 understanding.")
    parser.add_argument("--model-path", "-m", default=UNDERSTANDING_DIR.parent / "model_weight" / "MonkeyOCRv2-S-Und")
    parser.add_argument("--tensor-parallel-size", "--tp", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    parser.add_argument("--max-model-len", type=int, default=8196)
    parser.add_argument("--max-num-seqs", type=int, default=128)
    parser.add_argument("--max-num-batched-tokens", type=int, default=16384)
    parser.add_argument("--served-model-name", default="MonkeyOCRv2")
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", "-p", type=int, default=8889)
    parser.add_argument("extra_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    model_path = Path(args.model_path).expanduser()
    if not model_path.exists():
        raise SystemExit(f"Model path does not exist: {model_path}")
    ensure_port_available(args.host, args.port)
    argv = ["vllm", "serve", str(model_path), "--tensor-parallel-size", str(args.tensor_parallel_size),
            "--gpu-memory-utilization", str(args.gpu_memory_utilization), "--max-model-len", str(args.max_model_len),
            "--max-num-seqs", str(args.max_num_seqs), "--max-num-batched-tokens", str(args.max_num_batched_tokens),
            "--served-model-name", args.served_model_name, "--port", str(args.port), "--trust-remote-code"]
    if args.host:
        argv += ["--host", args.host]
    if args.extra_args:
        argv.extend(args.extra_args[1:] if args.extra_args[0] == "--" else args.extra_args)
    print("Running:", " ".join(argv))
    sys.argv = argv
    vllm_main()


if __name__ == "__main__":
    main()
