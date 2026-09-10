"""Offscreen rendering with per-process context reuse (spec 17.4.2).

Why this exists rather than ``mujoco.Renderer``
-----------------------------------------------
Every scene compiles a fresh ``MjModel`` (geom types, distractor presence, and
occluder presence all vary), and ``mujoco.Renderer`` binds to a model, so the
naive loop constructs one per scene. Measured on this workload:

    MJCF compile        0.29 ms
    1600 physics steps  5.88 ms
    10 renders          5.64 ms
    mujoco.Renderer()   8.22 ms   <- larger than physics and rendering combined

Of that 8.22 ms, 6.21 ms is ``MjvScene`` allocating its default 10,000-geom
buffer for scenes that contain five geoms, and 1.42 ms is creating an OS-level
GL context. Neither depends on scene content.

This renderer keeps one GL context per process and rebuilds only the
model-dependent pieces, sized to the model's actual geom count. Per-scene
overhead drops from 8.22 ms to about 0.7 ms.

Segmentation rendering deliberately still goes through ``mujoco.Renderer``: it
is used only to validate the ray-cast occlusion estimator, never on the
generation path, so there is nothing to gain from reimplementing its id decoding
and a correctness risk in doing so.
"""

from __future__ import annotations

import numpy as np

import mujoco

# Re-exported so existing call sites keep working; the implementations live in
# physground.frames, which imports no mujoco.
from .frames import PNG_COMPRESS_LEVEL, pack_frames, unpack_frames  # noqa: F401

__all__ = ["SceneRenderer", "pack_frames", "unpack_frames", "PNG_COMPRESS_LEVEL"]

_FONT_SCALE = mujoco.mjtFontScale.mjFONTSCALE_150


class SceneRenderer:
    """Renders RGB frames from a succession of models, reusing what it can.

    Not thread-safe and not shareable across processes: a GL context belongs to
    the thread that made it current. The pipeline runs one process per core
    (spec 17.4.4), which fits this exactly.
    """

    def __init__(self, height: int = 224, width: int = 224, max_geom: int | None = None):
        self._height = int(height)
        self._width = int(width)
        self._max_geom = max_geom
        self._rect = mujoco.MjrRect(0, 0, self._width, self._height)

        # One OS-level GL context for the life of the process.
        self._gl = mujoco.GLContext(self._width, self._height)
        self._gl.make_current()

        self._model: mujoco.MjModel | None = None
        self._mjr: mujoco.MjrContext | None = None
        self._scene: mujoco.MjvScene | None = None
        self._option = mujoco.MjvOption()
        self._camera = mujoco.MjvCamera()
        self._camera.type = mujoco.mjtCamera.mjCAMERA_FIXED
        self._rgb = np.empty((self._height, self._width, 3), dtype=np.uint8)

    # -- lifecycle ---------------------------------------------------------- #

    def _bind(self, model: mujoco.MjModel) -> None:
        if self._model is model:
            return
        if self._mjr is not None:
            self._mjr.free()
        self._mjr = mujoco.MjrContext(model, _FONT_SCALE)
        # Render into the offscreen buffer, whose size the MJCF fixes at exactly
        # the render resolution via <global offwidth/offheight> (spec 17.4.1).
        mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, self._mjr)
        # Headroom over the model's own geoms for the decorative geoms mjv adds
        # (contact points, force arrows) if visualisation flags are ever enabled.
        cap = self._max_geom if self._max_geom is not None else int(model.ngeom) + 64
        self._scene = mujoco.MjvScene(model, cap)
        self._model = model

    def close(self) -> None:
        if self._mjr is not None:
            self._mjr.free()
            self._mjr = None
        if self._gl is not None:
            self._gl.free()
            self._gl = None
        self._model = None
        self._scene = None

    def __enter__(self) -> "SceneRenderer":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- rendering ---------------------------------------------------------- #

    def render(self, model: mujoco.MjModel, data: mujoco.MjData, camera_id: int) -> np.ndarray:
        """Render one frame as ``(H, W, 3)`` uint8.

        Returns a fresh array each call; the internal read buffer is reused, so
        a caller that kept the returned view would see it overwritten by the
        next frame.
        """
        self._gl.make_current()
        self._bind(model)
        self._camera.fixedcamid = int(camera_id)

        mujoco.mjv_updateScene(model, data, self._option, None, self._camera,
                               mujoco.mjtCatBit.mjCAT_ALL, self._scene)
        mujoco.mjr_render(self._rect, self._scene, self._mjr)
        mujoco.mjr_readPixels(self._rgb, None, self._rect, self._mjr)
        # OpenGL reads bottom-up; image conventions are top-down.
        return np.flipud(self._rgb).copy()
