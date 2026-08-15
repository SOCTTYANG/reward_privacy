

from __future__ import annotations

import os
from pkgutil import extend_path


__path__ = extend_path(__path__, __name__)

_project_root = os.environ.get("SAFE_RLHF_PROJECT_ROOT") or os.environ.get("ROOT_DIR")
if not _project_root:
    _project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

_main_safe_rlhf_path = os.path.join(os.path.abspath(_project_root), "safe_rlhf")
if os.path.isdir(_main_safe_rlhf_path) and _main_safe_rlhf_path not in __path__:
    __path__.append(_main_safe_rlhf_path)

__all__: list[str] = []
