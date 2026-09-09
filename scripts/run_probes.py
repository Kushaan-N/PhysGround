#!/usr/bin/env python
"""Run one experiment (spec 10).

    python scripts/run_probes.py --exp A --seed 0
    python scripts/run_probes.py --exp D --seed 0,1,2,3,4

Experiments are ordered so each is a kill-switch for everything after it and
costs less than what follows. A and B exit non-zero on their kill conditions, so
a shell pipeline stops rather than spending the grid budget on a broken pipeline
(spec 10).

Only raw per-item predictions are written (spec 0.4); summaries come from
scripts/make_figures.py.
"""

from __future__ import annotations

import argparse
import json
import os

os.environ.setdefault("MUJOCO_GL", "osmesa")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--exp", required=True, help="A|B|C|D|E|F, or a comma-separated list")
    parser.add_argument("--seed", default="0",
                        help="probe seed(s); spec 3.5 asks for 5 per reported cell")
    parser.add_argument("--condition", default="base")
    parser.add_argument("--no-kill", action="store_true",
                        help="report kill conditions without exiting non-zero")
    args = parser.parse_args()

    import random          # noqa: E402
    import numpy as np     # noqa: E402

    from physground.experiments import EXPERIMENTS   # noqa: E402

    experiments = [e.strip().upper() for e in args.exp.split(",")]
    seeds = [int(s) for s in args.seed.split(",")]

    failures = []
    for name in experiments:
        if name not in EXPERIMENTS:
            raise SystemExit(f"unknown experiment {name!r}; choose from {sorted(EXPERIMENTS)}")
        for seed in seeds:
            random.seed(seed)
            np.random.seed(seed)
            kwargs = {"seed": seed}
            if name not in ("E",):
                kwargs["condition"] = args.condition
            report = EXPERIMENTS[name](**kwargs)
            print(json.dumps(report, indent=2, default=str), flush=True)
            if report.get("passed") is False:
                failures.append((name, seed, report.get("kill_condition", "")))

    if failures and not args.no_kill:
        lines = "\n".join(f"  Exp {n} seed {s}: {why}" for n, s, why in failures)
        raise SystemExit(f"kill condition met; do not proceed (spec 10):\n{lines}")


if __name__ == "__main__":
    main()
