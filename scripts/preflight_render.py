#!/usr/bin/env python
"""Verify the renderer is what we think it is, before generating anything (spec 17.3).

MuJoCo reads ``MUJOCO_GL`` once, at ``import mujoco``, so this module sets it
*before* that import and before importing anything from ``physground``. Setting
it afterwards silently has no effect (spec 14, 17.6).

The rule this encodes: software rendering is acceptable, but only when it is
*chosen*. On a CPU container that happens to have Mesa's EGL, ``MUJOCO_GL=egl``
succeeds and hands you llvmpipe with no error at all (spec 17.1, case 3) --
which is why checking that EGL initialised is not verification. Something
answered; the question is what. So this checks the renderer's *identity* and its
*timing*, because either alone can be fooled: a fast software rasteriser on an
idle machine can beat the timing bound, and an unrecognised vendor string can
pass the identity check.
"""

from __future__ import annotations

import os
import sys


def configure_gl(backend: str | None = None) -> str:
    """Settle ``MUJOCO_GL`` before mujoco is imported anywhere. Returns the choice.

    An explicit value always wins, whether passed here or already exported --
    the whole point of spec 17.1 is that the slow path must be *chosen*, so this
    never overrides a deliberate setting.

    The default is platform-dependent. ``osmesa`` is the right headless choice
    on Linux, where the cluster and Modal run, but MuJoCo rejects it outright on
    macOS, which uses CGL and needs no hint. Defaulting to osmesa everywhere
    made the documented quick-start fail on any Mac with an error about an
    environment variable rather than about rendering.
    """
    if backend:
        os.environ["MUJOCO_GL"] = backend
    elif "MUJOCO_GL" not in os.environ and sys.platform.startswith("linux"):
        os.environ["MUJOCO_GL"] = "osmesa"

    chosen = os.environ.get("MUJOCO_GL", "")
    if chosen == "osmesa":
        os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
    # MuJoCo's internal threading fights container-level parallelism; the
    # pipeline runs one process per core instead (spec 17.4.4).
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    return chosen or "(platform default)"


# MUST run before `import mujoco`, and therefore before any physground import.
configure_gl()

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

_MJCF = """
<mujoco>
  <visual>
    <global offwidth="224" offheight="224"/>
    <quality shadowsize="0" offsamples="0"/>
  </visual>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1" castshadow="false"/>
    <geom type="plane" size="1 1 0.1" rgba="0.8 0.8 0.8 1"/>
    <body pos="0 0 0.1"><freejoint/>
      <geom type="box" size="0.05 0.05 0.05" mass="0.5" rgba="0.9 0.2 0.2 1"/>
    </body>
  </worldbody>
</mujoco>
"""

_SOFTWARE_MARKERS = ("llvmpipe", "softpipe", "swrast", "lavapipe")


def preflight(warn_ms: float = 15.0, fail_ms: float = 60.0, n: int = 20,
              strict: bool = True) -> dict:
    """Render a known scene and report which backend actually did it."""
    model = mujoco.MjModel.from_xml_string(_MJCF)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    renderer = mujoco.Renderer(model, height=224, width=224)
    try:
        renderer.update_scene(data)
        renderer.render()                              # warm up; excluded from timing

        start = time.perf_counter()
        for _ in range(n):
            renderer.update_scene(data)
            frame = renderer.render()
        ms = (time.perf_counter() - start) / n * 1000.0

        if frame.shape != (224, 224, 3):
            raise RuntimeError(f"bad frame shape {frame.shape}")
        # A context that initialised but drew nothing returns a uniform buffer.
        # The scene contains a red box on a grey plane, so real output cannot be
        # flat -- this catches "rendering succeeded" with an empty framebuffer.
        if float(np.asarray(frame).std()) <= 1.0:
            raise RuntimeError("frame is blank - renderer produced no geometry")

        gl_renderer = "unknown"
        try:
            from OpenGL import GL
            value = GL.glGetString(GL.GL_RENDERER)
            if value is not None:
                gl_renderer = value.decode()
        except Exception:
            pass
    finally:
        renderer.close()

    backend = os.environ.get("MUJOCO_GL", "(platform default)")
    software = any(marker in gl_renderer.lower() for marker in _SOFTWARE_MARKERS)

    report = {"backend": backend, "gl_renderer": gl_renderer,
              "ms_per_frame": round(ms, 3), "software": bool(software)}
    print(f"[preflight] MUJOCO_GL={backend}  GL_RENDERER={gl_renderer}  "
          f"{ms:.1f} ms/frame  software={software}", flush=True)

    if strict and backend == "egl" and (software or ms > warn_ms):
        raise RuntimeError(
            f"MUJOCO_GL=egl but rendering looks like software "
            f"(GL_RENDERER={gl_renderer}, {ms:.1f} ms/frame). No NVIDIA EGL vendor ICD "
            "in this container. Either attach a GPU or set MUJOCO_GL=osmesa explicitly, "
            "so the slow path is a deliberate choice and the wall-clock estimates "
            "downstream stay valid.")
    if strict and ms > fail_ms:
        raise RuntimeError(f"rendering unusably slow: {ms:.1f} ms/frame")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warn-ms", type=float, default=15.0,
                        help="above this, an egl backend is treated as a software fallback")
    parser.add_argument("--fail-ms", type=float, default=60.0)
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--no-strict", action="store_true",
                        help="report without raising; for diagnosing a node, not for job use")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    report = preflight(args.warn_ms, args.fail_ms, args.n, strict=not args.no_strict)
    if args.json:
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
