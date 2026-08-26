"""Lock BackendConfig.server_url validation (task 1 / PR #27 recovery)."""

from __future__ import annotations

import pytest

from core_runner import BackendConfig, normalize_server_url


def test_empty_server_url_raises():
    with pytest.raises(ValueError, match="server_url is required"):
        BackendConfig(model_path="dummy")


def test_whitespace_server_url_raises():
    with pytest.raises(ValueError, match="server_url is required"):
        BackendConfig(model_path="dummy", server_url="   \n\t  ")


def test_valid_http_url_is_accepted():
    cfg = BackendConfig(model_path="dummy", server_url="http://127.0.0.1:8000")
    assert cfg.server_url == "http://127.0.0.1:8000"


def test_normalize_server_url_strips_and_defaults_http():
    assert normalize_server_url("  127.0.0.1:8000/  ") == "http://127.0.0.1:8000"
    assert normalize_server_url("") == ""
    assert normalize_server_url("   ") == ""
