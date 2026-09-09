#!/usr/bin/env python
"""Validate the ray-cast occlusion estimator against the double render (spec 17.4).

    python scripts/validate_occlusion.py --n-scenes 12

Spec 17.4 asks for Pearson r > 0.95 against the segmentation method before
substituting the ray cast in corpus generation. This runs that check and records
which method produced the corpus values, as spec 17.4 also requires.

The object is swept along its push ray rather than measured at one settled pose.
Every scene read at the same pose gives nearly the same occlusion, and a
correlation computed over a range of 0.04 would be measuring noise while looking
like agreement.
"""

from __future__ import annotations

import argparse
import json
import os

os.environ.setdefault("MUJOCO_GL", "osmesa")
if os.environ["MUJOCO_GL"] == "osmesa":
    os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")

import numpy as np  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-scenes", type=int, default=12)
    parser.add_argument("--n-samples", type=int, default=384,
                        help="surface points per ray-cast estimate")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.95)
    args = parser.parse_args()

    import mujoco                                          # noqa: E402
    from physground import factors as F                    # noqa: E402
    from physground import ground_truth as G               # noqa: E402
    from physground import paths as P                      # noqa: E402
    from physground import scene as S                      # noqa: E402

    ray, reference, labels = [], [], []
    for index in range(args.n_scenes):
        factors = F.sample_factors(index, args.seed)
        factors["spawn_height"] = 0.0
        model = mujoco.MjModel.from_xml_string(S.build_mjcf(factors, occluded=True))
        data = mujoco.MjData(model)
        ids = G.resolve_ids(model)
        angle = factors["approach_angle"]

        renderer = mujoco.Renderer(model, *S.RENDER_HW)
        try:
            renderer.enable_segmentation_rendering()
            for offset in (-0.06, -0.02, 0.02, 0.06, 0.10, 0.16):
                mujoco.mj_resetDataKeyframe(model, data, 0)
                radius = factors["obj_radius"] + offset
                data.qpos[ids.target_qpos] = radius * np.cos(angle)
                data.qpos[ids.target_qpos + 1] = radius * np.sin(angle)
                mujoco.mj_forward(model, data)
                ray.append(G.occlusion_fraction(model, data, ids, args.n_samples,
                                                np.random.default_rng(index)))
                reference.append(G.occlusion_fraction_segmentation(renderer, model, data, ids))
                labels.append(f"{index:04d}@{offset:+.2f}")
        finally:
            renderer.close()

    ray, reference = np.asarray(ray), np.asarray(reference)
    correlation = float(np.corrcoef(ray, reference)[0, 1])
    report = {
        "n_measurements": int(ray.size),
        "pearson_r": correlation,
        "mean_abs_difference": float(np.abs(ray - reference).mean()),
        "max_abs_difference": float(np.abs(ray - reference).max()),
        "ray_mean": float(ray.mean()), "ray_range": [float(ray.min()), float(ray.max())],
        "reference_mean": float(reference.mean()),
        "reference_range": [float(reference.min()), float(reference.max())],
        "threshold": args.threshold,
        "passed": bool(correlation > args.threshold),
        "corpus_method": "raycast_surface_weighted",
    }

    print(f"n={report['n_measurements']}  Pearson r={correlation:.4f} "
          f"(threshold {args.threshold})  mean|diff|={report['mean_abs_difference']:.4f}")
    print(f"ray       mean={report['ray_mean']:.3f} range={report['ray_range']}")
    print(f"reference mean={report['reference_mean']:.3f} range={report['reference_range']}")

    out_dir = P.gates_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    P.atomic_write_json(out_dir / "occlusion_validation.json", report)
    print(json.dumps({"written": str(out_dir / "occlusion_validation.json")}))

    if not report["passed"]:
        raise SystemExit(
            f"ray-cast estimator only reaches r={correlation:.3f}; spec 17.4 requires "
            f"> {args.threshold} before it may replace the double render.")


if __name__ == "__main__":
    main()
