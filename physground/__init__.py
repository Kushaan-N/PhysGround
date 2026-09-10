"""PhysGround: what physical state do frozen visual encoders actually encode?

Import discipline
-----------------
This module deliberately imports nothing heavy. MuJoCo reads ``MUJOCO_GL`` exactly
once, at ``import mujoco``, and setting it afterwards silently does nothing
(spec 14, 17.6). Entry scripts therefore set the environment variable first and
import project modules second; if ``physground/__init__.py`` pulled in
``physground.scene`` (and through it ``mujoco``) at package-import time, that
ordering would be impossible to honour and the failure would be invisible.

Submodules are exposed lazily through ``__getattr__`` so ``physground.scene``
still works as an attribute access without eagerly importing anything.
"""

from __future__ import annotations

import importlib
from typing import Any

__version__ = "0.1.0"

_LAZY = {
    "encoders",
    "experiments",
    "factors",
    "features",
    "figures",
    "gates",
    "ground_truth",
    "latent_dynamics",
    "paths",
    "probes",
    "render",
    "rollout",
    "scene",
    "splits",
    "stats",
    "summarize",
}

__all__ = sorted(_LAZY) + ["__version__"]


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        module = importlib.import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | _LAZY)
