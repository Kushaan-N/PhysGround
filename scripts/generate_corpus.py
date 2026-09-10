#!/usr/bin/env python
"""Generate the MuJoCo corpus (spec 5, 10 Exp 0).

    python scripts/generate_corpus.py --condition base     --n 3000 --seed 0
    python scripts/generate_corpus.py --condition occluded --n 1000 --seed 0

Occluded scenes reuse the sampled factors of the base scene with the same index
(spec 5.4), so `--n 1000` on the occluded condition pairs with the first 1000
base scenes automatically; no flag couples them.

MUJOCO_GL is read once at `import mujoco`, so it is resolved here before any
project import (spec 14, 17.6).
"""

from __future__ import annotations

import argparse


# Shared with preflight_render so both entry points make the same choice, and
# so the platform default lives in exactly one place.
from preflight_render import configure_gl  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--condition", choices=("base", "occluded"), default="base")
    parser.add_argument("--n", type=int, required=True, help="number of scenes in this condition")
    parser.add_argument("--seed", type=int, default=0, help="master seed (spec 0.5)")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--n-shards", type=int, default=1)
    parser.add_argument("--gl", default=None,
                        help="MUJOCO_GL backend. Default: osmesa on Linux, the platform's own on macOS.")
    parser.add_argument("--skip-preflight", action="store_true",
                        help="skip the render backend check. Only for a machine already verified.")
    parser.add_argument("--force", action="store_true", help="regenerate scenes that already exist")
    parser.add_argument("--progress-every", type=int, default=50)
    args = parser.parse_args()

    configure_gl(args.gl)

    if not args.skip_preflight:
        from preflight_render import preflight          # noqa: E402
        preflight()                                     # spec 17.3: every worker, every time

    import time                                          # noqa: E402
    from physground import paths as P                    # noqa: E402
    from physground.render import SceneRenderer          # noqa: E402
    from physground.rollout import SceneInvalid, generate_scene, scene_is_complete  # noqa: E402

    indices = list(range(args.shard, args.n, args.n_shards))
    print(f"[generate] condition={args.condition} shard {args.shard}/{args.n_shards}: "
          f"{len(indices)} scenes, seed {args.seed}", flush=True)

    # One renderer for the whole process, not one per scene (spec 17.4.2).
    renderer = SceneRenderer(224, 224)
    made = skipped = failed = 0
    started = time.perf_counter()
    try:
        for position, index in enumerate(indices):
            out_dir = P.scene_dir(args.condition, index)
            if not args.force and scene_is_complete(out_dir, index, args.seed, args.condition):
                skipped += 1
                continue
            try:
                generate_scene(index, args.seed, args.condition, out_dir, renderer)
                made += 1
            except SceneInvalid as exc:
                # A scene with no contact has no frames 3-6 and would contribute
                # ten silent negatives to the contact probe. Drop it and record
                # the rate rather than emitting a mislabelled scene.
                failed += 1
                print(f"[generate] scene {index} invalid: {exc}", flush=True)
            if args.progress_every and position % args.progress_every == 0:
                rate = (position + 1) / max(time.perf_counter() - started, 1e-9)
                print(f"[generate] {position + 1}/{len(indices)}  {rate:.1f} scenes/s", flush=True)
    finally:
        renderer.close()

    elapsed = time.perf_counter() - started
    print(f"[generate] done: {made} generated, {skipped} already present, {failed} invalid, "
          f"{elapsed:.0f}s ({made / max(elapsed, 1e-9):.1f} scenes/s)", flush=True)
    if failed and made and failed / (made + failed) > 0.02:
        raise SystemExit(
            f"{failed / (made + failed):.1%} of scenes produced no contact. That is above the 2% "
            "tolerance and suggests a trajectory or geometry problem, not bad luck.")


if __name__ == "__main__":
    main()
