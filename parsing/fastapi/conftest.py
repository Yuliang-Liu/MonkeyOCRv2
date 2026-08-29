"""Collect FastAPI contract tests without torch, GPU, or model weights.

``test_parse_api.py`` imports ``main``, which imports ``core_runner``. Reuse the
torch / preprocessor stubs from ``parsing/tests/conftest.py`` instead of copying
that suite, and put this directory on ``sys.path`` so ``import main`` works from
the repository root.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

FASTAPI_DIR = Path(__file__).resolve().parent
_TESTS_CONFTEST = FASTAPI_DIR.parent / "tests" / "conftest.py"

_spec = importlib.util.spec_from_file_location("_parsing_tests_stubs", _TESTS_CONFTEST)
if _spec is None or _spec.loader is None:
    raise ImportError(f"Could not load stub helpers from {_TESTS_CONFTEST}")
_stubs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_stubs)
_stubs._install_torch_stub_if_missing()
_stubs._install_preprocessor_stub()

if str(FASTAPI_DIR) not in sys.path:
    sys.path.insert(0, str(FASTAPI_DIR))
