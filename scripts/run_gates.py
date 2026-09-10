#!/usr/bin/env python
"""Run gates G1-G4 on the pilot (spec 9, 10 Exp 0).

    python scripts/run_gates.py --pilot-scenes 50 --corpus-scenes 3000 --seed 0

Spec 0.3: all four must pass before the full corpus is generated. G2 has no
automated verdict -- it writes a contact sheet a human must open (spec 17.6).
"""

from __future__ import annotations

import argparse

# Settled before any import that could reach mujoco (spec 14, 17.6). The gates
# read frames back rather than rendering, but gate_g4 re-extracts features and
# the import graph is easier to reason about with one rule.
from preflight_render import configure_gl  # noqa: E402

configure_gl()

import json  # noqa: E402

import numpy as np  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pilot-scenes", type=int, default=50,
                        help="scenes to draw label and visual checks from")
    parser.add_argument("--corpus-scenes", type=int, default=3000,
                        help="size at which G1 evaluates the sampler (spec 9.1 note)")
    parser.add_argument("--gates", default="1,2,3,4", help="comma-separated subset")
    parser.add_argument("--skip-g4", action="store_true",
                        help="G4 needs a GPU and re-extracts features; skip on a CPU node")
    args = parser.parse_args()

    from physground import features as feature_module   # noqa: E402
    from physground import gates as gate_module          # noqa: E402
    from physground import paths as P                    # noqa: E402
    from physground.figures import decorrelation_heatmap  # noqa: E402

    wanted = {g.strip() for g in args.gates.split(",")}
    out_dir = P.gates_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict] = {}

    scenes = feature_module.scene_indices_for("base")[: args.pilot_scenes]
    targets = feature_module.load_targets("base", scenes) if scenes else None

    if "1" in wanted:
        derived = None
        if targets is not None:
            derived = {k: targets[k] for k in
                       ("contact_state", "obj_pos_x", "obj_pos_y", "obj_speed", "occlusion_fraction")
                       if k in targets}
        report = gate_module.gate_g1(
            args.corpus_scenes, args.seed, out_dir=out_dir,
            derived=derived, scene_index=targets["scene_index"] if targets is not None else None)
        decorrelation_heatmap(report, P.figures_dir() / "fig1_decorrelation.png")
        results["G1"] = {k: v for k, v in report.items() if k not in ("rho_matrix", "pairs")}
        print(f"[G1] passed={report['passed']}  max|rho|={report['max_abs_rho']:.3f} "
              f"(threshold {report['threshold']}) at {report['worst_pair']}, "
              f"{report['n_significant_after_holm']} significant after Holm, "
              f"n={report['n_scenes']} scenes", flush=True)

    if "2" in wanted:
        pairs = [("base", i) for i in scenes[:20]]
        pairs += [("occluded", i) for i in feature_module.scene_indices_for("occluded")[:20]]
        report = gate_module.gate_g2(pairs, out_dir / "contact_sheet.png", seed=args.seed)
        results["G2"] = report
        print(f"[G2] wrote {report['path']} - OPEN IT AND LOOK AT IT. "
              "No automated check substitutes (spec 9.2, 17.6).", flush=True)

    if "3" in wanted:
        if targets is None:
            raise SystemExit("G3 needs a generated corpus; run generate_corpus.py first")
        report = gate_module.gate_g3(targets, targets["scene_index"], out_dir=out_dir)
        results["G3"] = report
        print(f"[G3] passed={report['passed']}  "
              f"{report['n_frames']} frames / {report['n_scenes']} scenes", flush=True)
        for check in report["checks"]:
            print(f"      {check['name']:26s} {check['value']:>12.5g}  passed={check['passed']}",
                  flush=True)

    if "4" in wanted and not args.skip_g4:
        report = gate_module.gate_g4(out_dir=out_dir)
        results["G4"] = report
        print(f"[G4] passed={report['passed']}  R2 fp32={report['r2_float32']:.4f} "
              f"fp16={report['r2_float16']:.4f}  |diff|={report['abs_difference']:.5f} "
              f"(tolerance {report['tolerance']})", flush=True)

    P.atomic_write_json(out_dir / "gates_summary.json", results)

    verdicts = {name: r.get("passed") for name, r in results.items()}
    failed = [name for name, ok in verdicts.items() if ok is False]
    needs_human = [name for name, ok in verdicts.items() if ok is None]
    print(f"\n[gates] {json.dumps(verdicts)}", flush=True)
    if needs_human:
        print(f"[gates] {', '.join(needs_human)} needs human inspection before proceeding.",
              flush=True)
    if failed:
        raise SystemExit(f"gates failed: {failed}. Fix generation; do not proceed (spec 10).")


if __name__ == "__main__":
    main()
