import sys
from argparse import Namespace
from pathlib import Path

UNDERSTANDING_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(UNDERSTANDING_DIR))

from vllm_argv import build_vllm_argv  # noqa: E402


def _args(**overrides) -> Namespace:
    values = dict(
        model_path="/tmp/MonkeyOCRv2-S-Und",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.5,
        max_model_len=8196,
        max_num_seqs=128,
        max_num_batched_tokens=16384,
        served_model_name="MonkeyOCRv2",
        host=None,
        port=8889,
    )
    values.update(overrides)
    return Namespace(**values)


def test_importing_vllm_argv_does_not_import_vllm() -> None:
    assert "vllm" not in sys.modules


def test_default_argv_includes_port_max_model_len_and_trust_remote_code() -> None:
    argv = build_vllm_argv(_args())
    assert argv[:3] == ["vllm", "serve", "/tmp/MonkeyOCRv2-S-Und"]
    assert argv[argv.index("--port") + 1] == "8889"
    assert argv[argv.index("--max-model-len") + 1] == "8196"
    assert "--trust-remote-code" in argv
    assert "--host" not in argv


def test_default_argv_matches_serve_contract() -> None:
    assert build_vllm_argv(_args()) == [
        "vllm",
        "serve",
        "/tmp/MonkeyOCRv2-S-Und",
        "--tensor-parallel-size",
        "1",
        "--gpu-memory-utilization",
        "0.5",
        "--max-model-len",
        "8196",
        "--max-num-seqs",
        "128",
        "--max-num-batched-tokens",
        "16384",
        "--served-model-name",
        "MonkeyOCRv2",
        "--port",
        "8889",
        "--trust-remote-code",
    ]


def test_host_is_appended_when_provided() -> None:
    argv = build_vllm_argv(_args(host="127.0.0.1"))
    assert argv[-2:] == ["--host", "127.0.0.1"]
    assert "--trust-remote-code" in argv
