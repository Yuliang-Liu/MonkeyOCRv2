"""Import parsing.core_runner without GPU, vLLM, or model weights.

``core_runner.py`` is a CLI/FastAPI/Gradio module, not a package.  Tests add
``parsing/`` to ``sys.path`` the same way ``parse.py`` does.

Top-level imports include ``torch`` and ``modeling.modeling_preprocessor``.
The latter pulls cv2/numpy/torch.  Neither is used by the pure functions under
test; they are stubbed so this suite runs on CPU with Pillow + requests only.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

PARSING_DIR = Path(__file__).resolve().parents[1]


def _install_import_stubs() -> None:
    if "torch" not in sys.modules:
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

    if "modeling.modeling_preprocessor" not in sys.modules:
        modeling = sys.modules.setdefault("modeling", types.ModuleType("modeling"))
        preprocessor = types.ModuleType("modeling.modeling_preprocessor")

        class Preprocessor:  # noqa: D401 - import stub
            """Placeholder matching the name imported by core_runner."""

        preprocessor.Preprocessor = Preprocessor
        modeling.modeling_preprocessor = preprocessor
        sys.modules["modeling.modeling_preprocessor"] = preprocessor


_install_import_stubs()
if str(PARSING_DIR) not in sys.path:
    sys.path.insert(0, str(PARSING_DIR))
