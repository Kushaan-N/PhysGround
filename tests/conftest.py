"""Shared fixtures.

``MUJOCO_GL`` is resolved here, before any test imports mujoco. pytest collects
test modules in one process, so a single module that imported mujoco early would
fix the backend for the entire session (spec 14).
"""

from __future__ import annotations

import os
import sys

# Platform-aware default. `osmesa` is the right headless choice on Linux, where
# CI and the cluster run, but MuJoCo rejects it outright on macOS -- which uses
# CGL and needs no backend hint at all. Hard-coding osmesa here makes the whole
# suite fail to collect on a developer's Mac with an error about an environment
# variable rather than about anything under test.
if "MUJOCO_GL" not in os.environ and sys.platform != "darwin":
    os.environ["MUJOCO_GL"] = "osmesa"
if os.environ.get("MUJOCO_GL") == "osmesa":
    os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")

import tempfile  # noqa: E402

import pytest  # noqa: E402


@pytest.fixture(scope="session")
def data_root(tmp_path_factory):
    """Isolated PHYSGROUND_DATA for the session."""
    root = tmp_path_factory.mktemp("physground-data")
    os.environ["PHYSGROUND_DATA"] = str(root)
    return root


@pytest.fixture(scope="session")
def renderer():
    from physground.render import SceneRenderer
    r = SceneRenderer(224, 224)
    yield r
    r.close()


def pytest_addoption(parser):
    parser.addoption("--run-slow", action="store_true",
                     help="run tests that download checkpoints or generate many scenes")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-slow"):
        return
    skip = pytest.mark.skip(reason="needs --run-slow (downloads checkpoints / heavy generation)")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip)
