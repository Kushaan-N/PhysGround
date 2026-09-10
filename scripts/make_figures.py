#!/usr/bin/env python
"""Regenerate every table and figure from the raw prediction archives (spec 0.4, 15).

    python scripts/make_figures.py --seeds 0,1,2,3,4

Touches no model and no GPU. Every number is recomputed from the .npz files the
experiments wrote, so a statistic can be revised, a bootstrap re-seeded, or a
multiple-comparison correction re-applied without re-running anything expensive.
"""

from __future__ import annotations

import argparse
import json

def _verdict(test: dict, alpha: float = 0.05) -> str:
    """Combine the superiority test and the equivalence test into one reading.

    The two are not alternatives and both can fire at once. That case is the
    whole reason spec 11 asks for a pre-registered smallest effect size: a
    difference can be statistically distinguishable from the baseline and still
    be too small to matter. Reporting only the p-value there would call a
    negligible effect "above baseline"; reporting only the TOST would call a
    real one "equivalent". Naming the conjunction is the honest option, and it
    is the expected outcome for H2's targets at a large enough corpus.
    """
    superior = test.get("p_superiority_holm", 1.0) < alpha
    equivalent = bool(test["tost_equivalent"])
    if superior and equivalent:
        return f"detectable but below the {test['tost_margin']:g} effect-size threshold"
    if superior:
        return "above baseline"
    if equivalent:
        return "equivalent to baseline"
    return "inconclusive"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", default="0", help="comma-separated probe seeds")
    parser.add_argument("--n-boot", type=int, default=2000, help="cluster bootstrap iterations (spec 11)")
    parser.add_argument("--margin", type=float, default=0.05,
                        help="smallest effect size of interest for TOST; must match prereg.md")
    parser.add_argument("--exps", default="A,B,C,D,E,F")
    args = parser.parse_args()

    import numpy as np                                    # noqa: E402
    from physground import paths as P                     # noqa: E402
    from physground import summarize as S                 # noqa: E402
    from physground.features import PER_SCENE_TARGETS, TARGETS   # noqa: E402
    from physground.figures import (frame_vs_video_figure, main_result_figure,  # noqa: E402
                                    occlusion_figure)
    from physground.stats import holm                     # noqa: E402

    seeds = [int(s) for s in args.seeds.split(",")]
    wanted = [e.strip().upper() for e in args.exps.split(",")]
    figures_dir = P.figures_dir()
    tables_dir = P.data_root() / "tables"

    all_rows: list[dict] = []
    for exp in wanted:
        rows = []
        for seed in seeds:
            try:
                rows.extend(S.summarize_experiment(exp, seed, n_boot=args.n_boot))
            except FileNotFoundError:
                continue
        if not rows:
            print(f"[figures] no results for experiment {exp}; skipping", flush=True)
            continue
        S.rows_to_csv(rows, tables_dir / f"exp_{exp}_per_seed.csv")
        aggregated = S.aggregate_seeds(rows)
        S.rows_to_csv(aggregated, tables_dir / f"exp_{exp}_summary.csv")
        all_rows.extend(aggregated)
        print(f"[figures] experiment {exp}: {len(rows)} rows over {len(seeds)} seed(s) "
              f"-> {len(aggregated)} cells", flush=True)

    # ---- Figure 2: the main result -------------------------------------- #
    main_rows = [r for r in all_rows if r["exp"] in ("D", "C") and r["metric"] != "auroc"]
    if main_rows:
        for view in ("cls", "mean"):
            main_result_figure(main_rows, figures_dir / f"fig2_main_{view}.png", view=view)
        print(f"[figures] wrote fig2_main_*.png", flush=True)

    # ---- Figure 3: frame versus video (H3) ------------------------------- #
    video_rows = [r for r in all_rows if r["exp"] == "F"]
    if video_rows:
        frame_vs_video_figure(video_rows, figures_dir / "fig3_frame_vs_video.png",
                              targets=PER_SCENE_TARGETS)
        print("[figures] wrote fig3_frame_vs_video.png", flush=True)

    # ---- Figure 4: occlusion (H4) ---------------------------------------- #
    for seed in seeds:
        try:
            curve = S.occlusion_curve("E", seed, ("contact_state", "obj_pos_x"))
        except FileNotFoundError:
            break
        if curve["series"]:
            occlusion_figure(np.asarray(curve["bins"]), curve["series"],
                             figures_dir / "fig4_occlusion.png")
            P.atomic_write_json(tables_dir / f"occlusion_curve_seed{seed:02d}.json", curve)
            print("[figures] wrote fig4_occlusion.png", flush=True)
            break

    # ---- Hypothesis tests ------------------------------------------------ #
    tests = []
    for target in TARGETS:
        for seed in seeds:
            try:
                tests.append(S.compare_to_baseline("D", seed, "dinov2_b", "random_b", target,
                                                   margin=args.margin, n_boot=args.n_boot))
            except (FileNotFoundError, KeyError, RuntimeError):
                continue
    if tests:
        # Holm within each property's family. Spec 11 requires the family be
        # stated explicitly: it is the set of seeds tested for that one target.
        by_target: dict[str, list[dict]] = {}
        for test in tests:
            by_target.setdefault(test["target"], []).append(test)
        for target, group in by_target.items():
            adjusted = holm([t["p_superiority"] for t in group])
            for test, value in zip(group, adjusted):
                test["p_superiority_holm"] = float(value)
                test["holm_family"] = f"seeds tested for target={target}"
        S.rows_to_csv(tests, tables_dir / "hypothesis_tests.csv")
        print(f"[figures] wrote hypothesis_tests.csv ({len(tests)} comparisons)", flush=True)
        for test in tests:
            test["verdict"] = _verdict(test)
            print(f"      {test['target']:16s} diff={test['observed_difference']:+.4f} "
                  f"p_holm={test.get('p_superiority_holm', float('nan')):.4f} "
                  f"TOST={'yes' if test['tost_equivalent'] else 'no'}  -> {test['verdict']}",
                  flush=True)
        S.rows_to_csv(tests, tables_dir / "hypothesis_tests.csv")

    print(json.dumps({"figures": str(figures_dir), "tables": str(tables_dir),
                      "cells": len(all_rows)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
