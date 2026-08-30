"""Cover ``build_vllm_argv`` without importing vLLM or touching weights."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

from vllm_argv import build_vllm_argv


def _args(**overrides):
    values = dict(
        model_path="/models/MonkeyOCRv2-B-Parsing",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.5,
        max_model_len=16384,
        max_num_batched_tokens=16384,
        served_model_name="MonkeyOCRv2",
        port=8888,
        max_num_seqs=128,
        host=None,
        target_attention_backend=None,
        draft_model=None,
        dflash_num_speculative_tokens=16,
        dflash_attention_backend=None,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def test_importing_vllm_argv_does_not_load_vllm():
    assert not any(name == "vllm" or name.startswith("vllm.") for name in sys.modules)


def test_no_draft_omits_speculative_config():
    argv = build_vllm_argv(_args())
    assert argv[0] == "vllm"
    assert argv[1] == "serve"
    assert argv[2] == "/models/MonkeyOCRv2-B-Parsing"
    assert "--port" in argv
    assert argv[argv.index("--port") + 1] == "8888"
    assert "--trust-remote-code" in argv
    assert "--speculative-config" not in argv
    assert all("dflash" not in part for part in argv)


def test_trust_remote_code_is_last():
    argv = build_vllm_argv(_args())
    assert argv[-1] == "--trust-remote-code"
    argv = build_vllm_argv(_args(draft_model="~/draft-weights", dflash_num_speculative_tokens=8))
    assert argv[-1] == "--trust-remote-code"


def test_draft_model_emits_dflash_speculative_config():
    argv = build_vllm_argv(
        _args(draft_model="~/draft-weights", dflash_num_speculative_tokens=8)
    )
    config = json.loads(argv[argv.index("--speculative-config") + 1])
    assert config["method"] == "dflash"
    assert config["model"] == str(Path("~/draft-weights").expanduser())
    assert config["num_speculative_tokens"] == 8
    assert "attention_backend" not in config


def test_host_and_attention_backend_are_optional():
    without = build_vllm_argv(_args())
    assert "--host" not in without
    assert "--attention-backend" not in without

    with_both = build_vllm_argv(
        _args(host="127.0.0.1", target_attention_backend="FLASH_ATTN")
    )
    assert with_both[with_both.index("--host") + 1] == "127.0.0.1"
    assert with_both[with_both.index("--attention-backend") + 1] == "FLASH_ATTN"


def test_falsy_max_num_seqs_omits_flag():
    argv = build_vllm_argv(_args(max_num_seqs=0))
    assert "--max-num-seqs" not in argv
    argv = build_vllm_argv(_args(max_num_seqs=None))
    assert "--max-num-seqs" not in argv
    argv = build_vllm_argv(_args(max_num_seqs=128))
    assert argv[argv.index("--max-num-seqs") + 1] == "128"
