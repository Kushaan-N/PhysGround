"""Turn raw prediction archives into summary rows, tests, and figures (spec 0.4, 11).

Nothing here re-runs a model. Every number is recomputed from the ``.npz`` files
the experiments wrote, so a statistic can be revised, a bootstrap re-seeded, or a
correction re-applied without touching a GPU. That is the point of persisting
raw arrays rather than summaries.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from . import paths as P
from . import stats as stats_module
from .features import CLASSIFICATION_TARGETS, REGRESSION_TARGETS

__all__ = [
    "load_raw",
    "summarize_experiment",
    "aggregate_seeds",
    "compare_to_baseline",
    "occlusion_curve",
    "METRIC_FOR",
]

#: Headline metric per target. Balanced accuracy for classification (spec 8.2)
#: because chance stays 0.5 whatever the class balance; AUROC is reported
#: alongside rather than instead, since it ignores the decision threshold the
#: probe actually chose.
METRIC_FOR = {t: "r2" for t in REGRESSION_TARGETS}
METRIC_FOR.update({t: "balanced_accuracy" for t in CLASSIFICATION_TARGETS})


def _parse_key(key: str) -> tuple[str, ...] | None:
    """Split ``encoder|condition|Llayer|view|task|evalcond|target|field``."""
    parts = key.split("|")
    if len(parts) != 8:
        return None
    return tuple(parts)


def load_raw(exp: str, seed: int, root: Path | None = None) -> dict[str, np.ndarray]:
    directory = (Path(root) / "results" / exp / f"seed_{seed:02d}") if root else P.results_dir(exp, seed)
    path = directory / "raw.npz"
    if not path.exists():
        raise FileNotFoundError(f"no raw results at {path}; run scripts/run_probes.py --exp {exp}")
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def summarize_experiment(exp: str, seed: int, root: Path | None = None,
                         n_boot: int = stats_module.N_BOOT) -> list[dict]:
    """One row per (cell, evaluation condition, target) with a bootstrap CI."""
    arrays = load_raw(exp, seed, root)

    grouped: dict[tuple, dict[str, np.ndarray]] = defaultdict(dict)
    for key, value in arrays.items():
        parsed = _parse_key(key)
        if parsed is None:
            continue
        encoder, condition, layer, view, task, eval_condition, target, field = parsed
        grouped[(encoder, condition, layer, view, task, eval_condition, target)][field] = value

    rows: list[dict] = []
    for (encoder, condition, layer, view, task, eval_condition, target), fields in sorted(grouped.items()):
        if "y_true" not in fields or "scene_index" not in fields:
            continue
        y_true = fields["y_true"]
        scenes = fields["scene_index"]
        metric = METRIC_FOR.get(target, "r2")

        # One multiplicity matrix per cell, reused across that cell's metrics so
        # AUROC and balanced accuracy describe the same resampled scenes.
        multiplicities = stats_module.group_multiplicities(scenes, n_boot, seed)

        base = {"exp": exp, "seed": seed, "encoder": encoder, "train_condition": condition,
                "layer": int(layer.lstrip("L")), "view": view, "task": task,
                "eval_condition": eval_condition, "target": target,
                "n_rows": int(y_true.size), "n_scenes": int(np.unique(scenes).size)}

        if metric == "r2":
            result = stats_module.bootstrap_r2(y_true, fields["y_pred"], scenes,
                                               n_boot, seed, multiplicities)
            rows.append({**base, "metric": "r2", "value": result.value,
                         "ci_low": result.ci_low, "ci_high": result.ci_high,
                         "trivial": 0.0})
        else:
            result = stats_module.bootstrap_balanced_accuracy(
                y_true, fields["y_pred"], scenes, n_boot, seed, multiplicities)
            majority = float(max(np.mean(y_true), 1 - np.mean(y_true)))
            rows.append({**base, "metric": "balanced_accuracy", "value": result.value,
                         "ci_low": result.ci_low, "ci_high": result.ci_high,
                         "trivial": 0.5, "majority_rate": majority})
            if "y_score" in fields:
                auc = stats_module.bootstrap_auroc(y_true, fields["y_score"], scenes,
                                                   n_boot, seed, multiplicities)
                rows.append({**base, "metric": "auroc", "value": auc.value,
                             "ci_low": auc.ci_low, "ci_high": auc.ci_high, "trivial": 0.5,
                             "majority_rate": majority})
    return rows


def aggregate_seeds(rows: Iterable[dict]) -> list[dict]:
    """Average a cell across probe seeds (spec 3.5).

    The reported interval is the union of the per-seed bootstrap intervals, not
    the spread of the point estimates. Seeds differ only in which scenes land in
    the test set, so the spread across seeds understates uncertainty -- it omits
    the sampling error *within* each split, which is the larger term.
    """
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        key = (row["exp"], row["encoder"], row["train_condition"], row["layer"], row["view"],
               row["task"], row["eval_condition"], row["target"], row["metric"])
        grouped[key].append(row)

    out = []
    for key, group in sorted(grouped.items()):
        values = np.array([g["value"] for g in group], dtype=float)
        out.append({
            "exp": key[0], "encoder": key[1], "train_condition": key[2], "layer": key[3],
            "view": key[4], "task": key[5], "eval_condition": key[6], "target": key[7],
            "metric": key[8],
            "value": float(np.nanmean(values)),
            "ci_low": float(np.nanmin([g["ci_low"] for g in group])),
            "ci_high": float(np.nanmax([g["ci_high"] for g in group])),
            "seed_sd": float(np.nanstd(values)),
            "n_seeds": len(group),
            "trivial": group[0].get("trivial"),
            "majority_rate": group[0].get("majority_rate"),
        })
    return out


# --------------------------------------------------------------------------- #
# Hypothesis tests
# --------------------------------------------------------------------------- #

def compare_to_baseline(exp: str, seed: int, encoder: str, baseline: str, target: str,
                        root: Path | None = None, margin: float = 0.05,
                        n_boot: int = stats_module.N_BOOT,
                        baseline_exp: str | None = None) -> dict:
    """Paired comparison of an encoder against a baseline on one target.

    Returns both directions of evidence, because the paper needs both:

    * a one-sided bootstrap p-value for "the encoder beats the baseline", used
      with Holm correction for H1-style claims of presence;
    * a TOST equivalence verdict for "the difference is smaller than ``margin``",
      used for H2's claims of *absence* (spec 11).

    Both models are resampled over the *same* scenes via a shared multiplicity
    matrix. An unpaired difference of two independent bootstraps is much wider
    and would understate every contrast.
    """
    arrays = load_raw(exp, seed, root)
    baseline_arrays = load_raw(baseline_exp, seed, root) if baseline_exp else arrays

    encoder_fields = _best_cell(arrays, encoder, target)
    baseline_fields = _best_cell(baseline_arrays, baseline, target)
    if encoder_fields is None or baseline_fields is None:
        raise KeyError(f"missing cell for encoder={encoder!r} or baseline={baseline!r} "
                       f"on target {target!r}")

    scenes = encoder_fields["scene_index"]
    if not np.array_equal(scenes, baseline_fields["scene_index"]):
        raise RuntimeError(
            "encoder and baseline were evaluated on different rows, so a paired "
            "comparison is not defined. Both must come from the same probe seed.")

    metric = METRIC_FOR.get(target, "r2")
    multiplicities = stats_module.group_multiplicities(scenes, n_boot, seed)
    field = "y_pred"

    _, encoder_samples = stats_module.bootstrap_metric(
        metric, encoder_fields["y_true"], encoder_fields[field], scenes, n_boot, seed,
        multiplicities, return_samples=True)
    _, baseline_samples = stats_module.bootstrap_metric(
        metric, baseline_fields["y_true"], baseline_fields[field], scenes, n_boot, seed,
        multiplicities, return_samples=True)

    difference = encoder_samples - baseline_samples
    observed = (stats_module.METRIC_FUNCTIONS[metric](encoder_fields["y_true"], encoder_fields[field])
                - stats_module.METRIC_FUNCTIONS[metric](baseline_fields["y_true"], baseline_fields[field]))

    finite = difference[np.isfinite(difference)]
    p_superiority = float(np.mean(finite <= 0.0)) if finite.size else 1.0
    equivalence = stats_module.tost_bootstrap(difference, margin, observed=observed)

    return {"exp": exp, "seed": seed, "encoder": encoder, "baseline": baseline,
            "target": target, "metric": metric, "observed_difference": float(observed),
            "p_superiority": p_superiority, **{f"tost_{k}": v for k, v in equivalence.items()}}


def _best_cell(arrays: dict[str, np.ndarray], encoder: str, target: str) -> dict | None:
    """Highest-scoring (layer, view) for an encoder on a target.

    Selecting the best layer is a deliberate choice with a cost: it is a
    maximum over the grid, so its sampling distribution is optimistically
    biased. That is acceptable *because of the direction of the claims* -- H2
    argues a property is absent, and taking the encoder's best case makes that
    argument harder, not easier. Claims of presence go through Holm correction
    over the stated family instead.
    """
    best, best_value = None, -np.inf
    for key in arrays:
        parsed = _parse_key(key)
        if parsed is None:
            continue
        enc, _, _, _, _, _, tgt, field = parsed
        if enc != encoder or tgt != target or field != "y_true":
            continue
        prefix = key.rsplit("|", 1)[0]
        fields = {f: arrays[f"{prefix}|{f}"] for f in ("y_true", "y_pred", "scene_index")
                  if f"{prefix}|{f}" in arrays}
        if len(fields) < 3:
            continue
        metric = METRIC_FOR.get(target, "r2")
        value = stats_module.METRIC_FUNCTIONS[metric](fields["y_true"], fields["y_pred"])
        if np.isfinite(value) and value > best_value:
            best, best_value = fields, value
    return best


def occlusion_curve(exp: str, seed: int, targets: Sequence[str], root: Path | None = None,
                    bins: Sequence[float] = (0.0, 0.2, 0.4, 0.6, 1.01),
                    n_boot: int = 500, encoder: str = "dinov2_b") -> dict:
    """Paired degradation against occlusion fraction, for H4 (spec 15, figure 4).

    Each occluded frame is compared against **its own matched base frame** --
    same scene, same frame index, same sampled factors, same trajectory, same
    probe. Only the occluder differs (spec 5.4). Binning the occluded condition
    on its own, and normalising to whatever its least-occluded bin happened to
    score, throws that pairing away: the reference then carries its own sampling
    noise, and with roughly 17 scenes per bin the resulting curve is dominated
    by it. Measured that way the curve wandered between 0.54 and 1.29 with no
    trend, on data where the paired contrast is clean.

    Retained scale: within each bin, ``(occluded - chance) / (base - chance)``
    on the same rows. 1.0 means occlusion cost nothing; 0.0 means the probe fell
    to chance. Normalising is what lets R2 for position and balanced accuracy
    for contact share an axis, and the claim is about *relative* degradation
    anyway.
    """
    arrays = load_raw(exp, seed, root)
    bins = np.asarray(bins, dtype=float)
    series: dict[str, dict] = {}

    for target in targets:
        paired = _paired_rows(arrays, encoder, target)
        if paired is None:
            continue
        metric = METRIC_FOR.get(target, "r2")
        chance = 0.5 if metric != "r2" else 0.0
        # Occlusion varies within a scene as the object moves past the slab, so
        # binning on it also bins on *phase*. A narrow phase window contains
        # little positional variance, and an R2 whose denominator is the bin's
        # own variance then swings wildly for reasons that have nothing to do
        # with occlusion -- measured that way, obj_pos_y read -0.80 in the least
        # occluded bin and *improved* to +0.67 in the most occluded one. The
        # denominator is therefore fixed to the full matched set's variance, so
        # bins are on one scale and only the numerator responds to occlusion.
        global_ss_tot = float(np.sum((paired["y_true_occ"] - paired["y_true_occ"].mean()) ** 2)
                              / paired["y_true_occ"].size) if metric == "r2" else None

        def score(y_true, y_pred):
            if metric != "r2":
                return stats_module.METRIC_FUNCTIONS[metric](y_true, y_pred)
            if not global_ss_tot:
                return np.nan
            return 1.0 - float(np.mean((y_true - y_pred) ** 2)) / global_ss_tot

        values, lows, highs, base_values, counts = [], [], [], [], []
        for low, high in zip(bins[:-1], bins[1:]):
            mask = (paired["occlusion"] >= low) & (paired["occlusion"] < high)
            counts.append(int(mask.sum()))
            if mask.sum() < 30 or np.unique(paired["scenes"][mask]).size < 8:
                values.append(np.nan); lows.append(np.nan)
                highs.append(np.nan); base_values.append(np.nan)
                continue

            scenes = paired["scenes"][mask]
            base_point = score(paired["y_true_base"][mask], paired["y_pred_base"][mask])
            occ_point = score(paired["y_true_occ"][mask], paired["y_pred_occ"][mask])
            denominator = base_point - chance
            values.append(float((occ_point - chance) / denominator)
                          if abs(denominator) > 1e-9 else np.nan)
            base_values.append(float(base_point))

            # Bootstrap the ratio over scenes, keeping the pairing.
            multiplicities, index = stats_module.group_multiplicities(scenes, n_boot, seed)
            ratios = []
            for row in multiplicities:
                take = np.repeat(np.arange(scenes.size), row[index].astype(int))
                if take.size == 0:
                    continue
                b = score(paired["y_true_base"][mask][take], paired["y_pred_base"][mask][take])
                o = score(paired["y_true_occ"][mask][take], paired["y_pred_occ"][mask][take])
                if abs(b - chance) > 1e-9:
                    ratios.append((o - chance) / (b - chance))
            if ratios:
                lo, hi = np.percentile(np.asarray(ratios), [2.5, 97.5])
                lows.append(float(lo)); highs.append(float(hi))
            else:
                lows.append(np.nan); highs.append(np.nan)

        series[target] = {"value": values, "ci_low": lows, "ci_high": highs,
                          "base_value": base_values, "n_rows": counts, "metric": metric}
    return {"bins": bins.tolist(), "series": series, "encoder": encoder,
            "note": "paired: each occluded frame against its matched base frame"}


def _paired_rows(arrays: dict[str, np.ndarray], encoder: str, target: str) -> dict | None:
    """Align an occluded evaluation with its matched base evaluation.

    Both sides are stored sorted by ``(scene_index, frame_index)`` with a fixed
    frame count per scene, so restricting the base side to the scenes the
    occluded side covers makes the two row-for-row comparable. The row counts
    are checked rather than assumed -- a mismatch would silently compare
    different frames and produce a degradation number that means nothing.
    """
    base_prefix = _find_prefix(arrays, encoder, target, "base", task="transfer")
    occ_prefix = _find_prefix(arrays, encoder, target, "occluded", task="transfer")
    if base_prefix is None or occ_prefix is None:
        return None
    occlusion_key = f"{occ_prefix}|occlusion_fraction"
    if occlusion_key not in arrays:
        return None

    base_scenes = arrays[f"{base_prefix}|{target}|scene_index"]
    occ_scenes = arrays[f"{occ_prefix}|{target}|scene_index"]
    keep = np.isin(base_scenes, np.unique(occ_scenes))
    if int(keep.sum()) != occ_scenes.size:
        return None
    if not np.array_equal(base_scenes[keep], occ_scenes):
        return None

    return {
        "scenes": occ_scenes,
        "occlusion": arrays[occlusion_key],
        "y_true_occ": arrays[f"{occ_prefix}|{target}|y_true"],
        "y_pred_occ": arrays[f"{occ_prefix}|{target}|y_pred"],
        "y_true_base": arrays[f"{base_prefix}|{target}|y_true"][keep],
        "y_pred_base": arrays[f"{base_prefix}|{target}|y_pred"][keep],
    }


def _find_prefix(arrays, encoder: str, target: str, eval_condition: str,
                 task: str | None = None) -> str | None:
    """First cell prefix matching an encoder, target, evaluation condition, and task.

    ``task`` matters from Experiment E onward, where the same encoder appears
    twice per layer -- once trained on clean frames and once on occluded ones.
    Matching without it would pick whichever happened to be stored first and
    silently mix the transfer and matched-training analyses.
    """
    for key in arrays:
        parsed = _parse_key(key)
        if parsed is None:
            continue
        enc, _, _, _, cell_task, evalc, tgt, field = parsed
        if enc != encoder or tgt != target or evalc != eval_condition or field != "y_true":
            continue
        if task is not None and cell_task != task:
            continue
        return "|".join(parsed[:6])
    return None


def rows_to_csv(rows: Sequence[dict], path: Path) -> Path:
    """Write summary rows as CSV. Regenerable from raw arrays at any time."""
    if not rows:
        return Path(path)
    columns = list(rows[0].keys())
    lines = [",".join(columns)]
    for row in rows:
        lines.append(",".join("" if row.get(c) is None else str(row.get(c)) for c in columns))
    return P.atomic_write_bytes(Path(path), "\n".join(lines).encode("utf-8"))
