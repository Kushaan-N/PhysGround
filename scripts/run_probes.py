#!/usr/bin/env python
"""Run one experiment (spec 10).

    python scripts/run_probes.py --exp A,B --seed 0
    python scripts/run_probes.py --exp C,D,E --seed 0,1,2,3,4 --jobs 5

Experiments are ordered so each is a kill-switch for everything after it and
costs less than what follows. A and B exit non-zero on their kill conditions, so
a shell pipeline stops rather than spending the grid budget on a broken pipeline
(spec 10).

Only raw per-item predictions are written (spec 0.4); summaries come from
scripts/make_figures.py.

Resuming
--------
A run that dies partway through re-does nothing that finished. Each
(experiment, seed) is keyed by a config hash covering the probe seed and the
cell grid, and a completed archive matching that hash is skipped (spec 0.6).
Pass ``--force`` to redo regardless. Experiments A and B still recompute their
verdict from the cached archive rather than reporting "skipped" with no
verdict -- otherwise resuming a grid whose kill-switch had *failed* would sail
straight past it.
"""

from __future__ import annotations

import argparse
import json
import os


def _run_one(payload: tuple) -> dict:
    """Run one (experiment, seed). Top level so it survives pickling to a worker."""
    name, seed, condition, force = payload

    # Each worker gets one BLAS thread. Without this every worker tries to use
    # every core, and N workers oversubscribe the machine N-fold -- which is
    # slower than running serially, not faster.
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(var, "1")

    import random
    import numpy as np
    from physground.experiments import EXPERIMENTS

    random.seed(seed)
    np.random.seed(seed)

    kwargs = {"seed": seed, "force": force}
    if name != "E":                      # E spans both conditions by construction
        kwargs["condition"] = condition
    return EXPERIMENTS[name](**kwargs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--exp", required=True,
                        help="A|B|C|D|E|F|G|S, or a comma-separated list. G is the optional "
                             "H5 latent-dynamics probe; S is the spatial contact probe of "
                             "spec 8.3 and needs the patch-token cache.")
    parser.add_argument("--seed", default="0",
                        help="probe seed(s); spec 3.5 asks for 5 per reported cell")
    parser.add_argument("--condition", default="base")
    parser.add_argument("--jobs", type=int, default=1,
                        help="seeds to run in parallel within each experiment. Seeds are "
                             "independent, so this is a straight wall-clock win -- but each "
                             "worker holds its own copy of the feature matrix, so keep it at 1 "
                             "for experiment S, whose patch tokens are gigabytes per worker.")
    parser.add_argument("--force", action="store_true",
                        help="re-run even where a completed archive matches the config hash")
    parser.add_argument("--no-kill", action="store_true",
                        help="report kill conditions without exiting non-zero")
    args = parser.parse_args()

    from physground.experiments import EXPERIMENTS   # noqa: E402

    experiments = [e.strip().upper() for e in args.exp.split(",")]
    seeds = [int(s) for s in args.seed.split(",")]
    for name in experiments:
        if name not in EXPERIMENTS:
            raise SystemExit(f"unknown experiment {name!r}; choose from {sorted(EXPERIMENTS)}")

    jobs = max(1, min(args.jobs, len(seeds)))
    failures = []

    # Experiments run in the order given, seeds within one in parallel. Keeping
    # the outer loop serial is what preserves the kill-switch ordering of spec
    # 10: A and B must be able to stop the run before C-F spend the grid budget.
    for name in experiments:
        payloads = [(name, seed, args.condition, args.force) for seed in seeds]

        if jobs == 1:
            reports = [_run_one(p) for p in payloads]
        else:
            from concurrent.futures import ProcessPoolExecutor
            print(f"[probes] {name}: {len(seeds)} seeds across {jobs} workers", flush=True)
            with ProcessPoolExecutor(max_workers=jobs) as pool:
                reports = list(pool.map(_run_one, payloads))

        for report in reports:
            print(json.dumps(report, indent=2, default=str), flush=True)
            if report.get("passed") is False:
                failures.append((name, report.get("seed"), report.get("kill_condition", "")))

        if failures and not args.no_kill:
            break          # do not start the next experiment behind a failed gate

    if failures and not args.no_kill:
        lines = "\n".join(f"  Exp {n} seed {s}: {why}" for n, s, why in failures)
        raise SystemExit(f"kill condition met; do not proceed (spec 10):\n{lines}")


if __name__ == "__main__":
    main()
