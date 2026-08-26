"""Import ``core_runner`` without loading the model preprocessor or weights.

The tests exercise pure helpers only.  ``core_runner`` imports the preprocessor
at module import time, so replace that one module before the test modules load.
Keep a real torch installation intact when one is present; in a minimal CI
environment, provide only the small ``torch.cuda`` surface used by cleanup code.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

PARSING_DIR = Path(__file__).resolve().parents[1]


def _install_torch_stub_if_missing() -> None:
    try:
        __import__("torch")
        return
    except ModuleNotFoundError:
        pass

    torch = types.ModuleType("torch")

    class _Cuda:
        @staticmethod
        def is_available() -> bool:
            return False

        @staticmethod
        def empty_cache() -> None:
            return None

    torch.cuda = _Cuda()
    sys.modules["torch"] = torch


def _install_preprocessor_stub() -> None:
    modeling = sys.modules.get("modeling")
    if modeling is None:
        modeling = types.ModuleType("modeling")
        modeling.__path__ = [str(PARSING_DIR / "modeling")]
        sys.modules["modeling"] = modeling

    preprocessor = types.ModuleType("modeling.modeling_preprocessor")

    class Preprocessor:  # noqa: D401 - import stub
        """Placeholder matching the name imported by core_runner."""

    preprocessor.Preprocessor = Preprocessor
    modeling.modeling_preprocessor = preprocessor
    sys.modules["modeling.modeling_preprocessor"] = preprocessor


_install_torch_stub_if_missing()
_install_preprocessor_stub()
if str(PARSING_DIR) not in sys.path:
    sys.path.insert(0, str(PARSING_DIR))
