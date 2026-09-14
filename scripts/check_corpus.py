#!/usr/bin/env python
"""Confirm a generated corpus is complete and healthy, before spending GPU time.

    python scripts/check_corpus.py --condition base --expected 3000
    python scripts/check_corpus.py --condition occluded --expected 1000 --paired-with base

Corpus generation runs as N parallel shards. If one dies — preemption, OOM, a
node going away — the others finish and the run *looks* successful: there is no
single process whose exit code reports the hole. The gap then surfaces much
later as a feature matrix with fewer rows than expected, or as a quietly smaller
training set, neither of which points back at generation.

This is read-only, takes seconds, and needs no GPU. Run it after generation and
before extraction.

It is **not** Gate G3. G3 checks whether the *labels* are sane (class balance,
finiteness, sampling distributions). This checks whether the *corpus* is all
there, and surfaces the per-scene quality flags the rollout already records.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--condition", default="base", choices=("base", "occluded"))
    parser.add_argument("--expected", type=int, default=None,
                        help="scene count you asked for; missing indices are listed if set")
    parser.add_argument("--paired-with", default=None,
                        help="check every scene here has a counterpart in that condition "
                             "(spec 5.4 pairs occluded scenes with base by index)")
    parser.add_argument("--max-missing-listed", type=int, default=20)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    from physground import paths as P                    # noqa: E402
    from physground.features import scene_indices_for    # noqa: E402

    found = scene_indices_for(args.condition)
    report: dict = {"condition": args.condition, "found": len(found)}
    problems: list[str] = []

    if not found:
        raise SystemExit(f"no complete scenes under {P.corpus_dir(args.condition)}")

    # --- completeness --------------------------------------------------- #
    if args.expected is not None:
        missing = sorted(set(range(args.expected)) - set(found))
        report["expected"] = args.expected
        report["missing"] = len(missing)
        if missing:
            shown = missing[:args.max_missing_listed]
            tail = "" if len(missing) <= len(shown) else f" (+{len(missing) - len(shown)} more)"
            problems.append(f"{len(missing)} scenes missing, e.g. {shown}{tail}")
            # A whole dead shard shows up as an arithmetic progression. Naming
            # the stride tells you which shard to re-run rather than re-running
            # the lot.
            if len(missing) > 2:
                strides = Counter(np.diff(missing).tolist())
                stride, count = strides.most_common(1)[0]
                if count > len(missing) * 0.8:
                    problems.append(
                        f"missing indices are mostly stride {stride} apart, which is what one "
                        f"dead shard of --n-shards {stride} looks like. Re-run shard "
                        f"{missing[0] % stride} rather than the whole condition.")

    # --- pairing (spec 5.4) ---------------------------------------------- #
    if args.paired_with:
        partner = set(scene_indices_for(args.paired_with))
        unpaired = sorted(set(found) - partner)
        report["unpaired"] = len(unpaired)
        if unpaired:
            problems.append(
                f"{len(unpaired)} scenes have no counterpart in {args.paired_with!r}, "
                f"e.g. {unpaired[:args.max_missing_listed]}. Matched pairs are what make the "
                "H4 comparison paired; unpaired scenes are silently dropped by Experiment E.")

    # --- per-scene quality flags ----------------------------------------- #
    flags = Counter()
    contact_frames, occlusion, episode_seconds = [], [], []
    unreadable = []
    for index in found:
        directory = P.scene_dir(args.condition, index)
        try:
            record = P.read_json(directory / "factors.json")
        except Exception:
            unreadable.append(index)
            continue
        quality = record.get("quality", {})
        for key in ("distractor_touched", "settled_by_end", "all_frames_in_frame"):
            if key in quality:
                flags[key] += bool(quality[key])
        contact_frames.append(quality.get("n_contact_frames", np.nan))
        occlusion.append(quality.get("mean_occlusion", np.nan))
        episode_seconds.append(quality.get("episode_seconds", np.nan))

    n = len(found) - len(unreadable)
    if unreadable:
        problems.append(f"{len(unreadable)} scenes have an unreadable factors.json")

    if n:
        contact_rate = float(np.nanmean(contact_frames)) / 10.0
        report.update({
            "contact_positive_rate": round(contact_rate, 4),
            "mean_occlusion": round(float(np.nanmean(occlusion)), 4),
            "episode_seconds_mean": round(float(np.nanmean(episode_seconds)), 3),
            "distractor_touched": flags["distractor_touched"],
            "all_frames_in_frame": flags["all_frames_in_frame"],
            "settled_by_end": flags["settled_by_end"],
        })
        # These mirror the invariants the scene design is supposed to guarantee.
        if flags["distractor_touched"]:
            problems.append(
                f"{flags['distractor_touched']} scenes contacted the distractor. It is only a "
                "control for object presence if it is never touched (spec 5.1).")
        if flags["all_frames_in_frame"] < n:
            problems.append(
                f"{n - flags['all_frames_in_frame']} scenes had the target leave the frame. "
                "Object position is the positive control; those rows are not measuring it.")
        if not 0.25 <= contact_rate <= 0.55:
            problems.append(
                f"contact positive rate {contact_rate:.3f} is outside G3's 0.25-0.55 band.")

    report["problems"] = problems
    report["passed"] = not problems

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"[corpus] {args.condition}: {report['found']} complete scenes")
        for key in ("expected", "missing", "unpaired", "contact_positive_rate",
                    "mean_occlusion", "episode_seconds_mean", "distractor_touched",
                    "all_frames_in_frame", "settled_by_end"):
            if key in report:
                print(f"           {key:24s} {report[key]}")
        for problem in problems:
            print(f"  [problem] {problem}")
        print(f"[corpus] {'OK' if report['passed'] else 'PROBLEMS FOUND'}")

    if problems:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
