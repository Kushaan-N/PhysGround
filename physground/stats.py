"""Metrics, cluster bootstrap, Holm-Bonferroni, and equivalence testing (spec 11).

Frames within a scene are not independent -- they share lighting, object
identity, colour, mass, and friction -- so every interval here resamples
*scenes* with replacement, never frames. Resampling frames would produce
intervals several times too narrow and would do so most severely for the
per-scene properties whose absence is the paper's claim.

Speed
-----
The naive cluster bootstrap rebuilds an index array of every frame on every one
of 2,000 iterations. For R2 and balanced accuracy that is unnecessary: both
decompose into per-scene sufficient statistics, so a whole bootstrap becomes one
matrix product against a multiplicity matrix. Resampling scenes with replacement
gives multiplicities that are exactly multinomial, which is what makes the
substitution exact rather than approximate.

AUROC does not decompose -- it depends on the joint ordering of all scores -- so
it is computed by weighting, in chunks, over a single global sort.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Callable, Sequence

import numpy as np

__all__ = [
    "BootstrapResult",
    "r2_score",
    "balanced_accuracy",
    "auroc",
    "group_multiplicities",
    "bootstrap_r2",
    "bootstrap_balanced_accuracy",
    "bootstrap_auroc",
    "bootstrap_metric",
    "holm",
    "tost_bootstrap",
    "METRIC_FUNCTIONS",
]

N_BOOT = 2000


@dataclass(frozen=True)
class BootstrapResult:
    value: float
    ci_low: float
    ci_high: float
    n_boot: int
    metric: str

    def as_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------- #
# Point metrics
# --------------------------------------------------------------------------- #

def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of determination against the mean of ``y_true``.

    Uses the *evaluation set's* own mean, which is the convention that makes
    R2 = 0 correspond to the trivial baseline of spec 8.4.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    if ss_tot <= 0.0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean of per-class recall. Chance is 0.5 regardless of class imbalance."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    recalls = []
    for cls in (0, 1):
        mask = y_true == cls
        if mask.any():
            recalls.append(float((y_pred[mask] == cls).mean()))
    return float(np.mean(recalls)) if recalls else float("nan")


def auroc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Area under the ROC curve, via the Mann-Whitney statistic with midranks."""
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)
    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = _midranks(scores[order])
    positive_rank_sum = float(ranks[y_true[order] == 1].sum())
    return (positive_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def _midranks(sorted_values: np.ndarray) -> np.ndarray:
    n = sorted_values.size
    starts = np.flatnonzero(np.r_[True, sorted_values[1:] != sorted_values[:-1]])
    counts = np.diff(np.r_[starts, n])
    return np.repeat(starts + (counts + 1) / 2.0, counts)


METRIC_FUNCTIONS: dict[str, Callable[[np.ndarray, np.ndarray], float]] = {
    "r2": r2_score,
    "balanced_accuracy": balanced_accuracy,
    "auroc": auroc,
}


# --------------------------------------------------------------------------- #
# Cluster bootstrap
# --------------------------------------------------------------------------- #

def group_multiplicities(groups: np.ndarray, n_boot: int = N_BOOT,
                         seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Multiplicity matrix for resampling scenes with replacement.

    Returns ``(multiplicities, group_index)`` where ``multiplicities`` is
    ``(n_boot, n_groups)`` and ``group_index`` maps each row to its group's
    column. Drawing ``n_groups`` scenes with replacement is exactly a multinomial
    draw over the scenes, so sampling the counts directly is equivalent to
    sampling the labels and far cheaper.

    Callers comparing two models must share one matrix so the comparison is
    paired on scenes; an unpaired difference of two independent bootstraps has a
    much wider spread and would understate every contrast in the paper.
    """
    groups = np.asarray(groups)
    unique, group_index = np.unique(groups, return_inverse=True)
    n_groups = unique.size
    rng = np.random.default_rng(seed)
    multiplicities = rng.multinomial(n_groups, np.full(n_groups, 1.0 / n_groups), size=n_boot)
    return multiplicities.astype(np.float64), group_index


def _percentile_ci(samples: np.ndarray, value: float, metric: str) -> BootstrapResult:
    finite = samples[np.isfinite(samples)]
    if finite.size == 0:
        return BootstrapResult(value, float("nan"), float("nan"), 0, metric)
    low, high = np.percentile(finite, [2.5, 97.5])
    return BootstrapResult(float(value), float(low), float(high), int(finite.size), metric)


def bootstrap_r2(y_true: np.ndarray, y_pred: np.ndarray, groups: np.ndarray,
                 n_boot: int = N_BOOT, seed: int = 0,
                 multiplicities: tuple[np.ndarray, np.ndarray] | None = None,
                 return_samples: bool = False):
    """Cluster-bootstrap CI for R2, via per-scene sufficient statistics.

    R2 needs only four per-scene sums -- count, residual sum of squares, sum of
    y, and sum of y squared -- so a whole bootstrap reduces to four matrix
    products. Both the residual and total sums of squares are recomputed under
    each resample; holding the total fixed at its observed value would treat the
    denominator as known and give intervals that are too narrow.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mult, index = multiplicities or group_multiplicities(groups, n_boot, seed)
    n_groups = mult.shape[1]

    counts = np.bincount(index, minlength=n_groups).astype(float)
    residual = np.bincount(index, weights=(y_true - y_pred) ** 2, minlength=n_groups)
    sum_y = np.bincount(index, weights=y_true, minlength=n_groups)
    sum_y2 = np.bincount(index, weights=y_true ** 2, minlength=n_groups)

    total_n = mult @ counts
    ss_res = mult @ residual
    total_y = mult @ sum_y
    ss_tot = (mult @ sum_y2) - np.divide(total_y ** 2, total_n,
                                         out=np.full_like(total_y, np.nan), where=total_n > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        samples = 1.0 - ss_res / ss_tot
    samples[~np.isfinite(ss_tot) | (ss_tot <= 0)] = np.nan

    result = _percentile_ci(samples, r2_score(y_true, y_pred), "r2")
    return (result, samples) if return_samples else result


def bootstrap_balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray, groups: np.ndarray,
                                n_boot: int = N_BOOT, seed: int = 0,
                                multiplicities: tuple[np.ndarray, np.ndarray] | None = None,
                                return_samples: bool = False):
    """Cluster-bootstrap CI for balanced accuracy, via per-scene confusion counts."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    mult, index = multiplicities or group_multiplicities(groups, n_boot, seed)
    n_groups = mult.shape[1]

    def counts(mask: np.ndarray) -> np.ndarray:
        return np.bincount(index, weights=mask.astype(float), minlength=n_groups)

    tp = mult @ counts((y_true == 1) & (y_pred == 1))
    fn = mult @ counts((y_true == 1) & (y_pred == 0))
    tn = mult @ counts((y_true == 0) & (y_pred == 0))
    fp = mult @ counts((y_true == 0) & (y_pred == 1))

    with np.errstate(divide="ignore", invalid="ignore"):
        tpr = np.where(tp + fn > 0, tp / (tp + fn), np.nan)
        tnr = np.where(tn + fp > 0, tn / (tn + fp), np.nan)
    samples = 0.5 * (tpr + tnr)

    result = _percentile_ci(samples, balanced_accuracy(y_true, y_pred), "balanced_accuracy")
    return (result, samples) if return_samples else result


def bootstrap_auroc(y_true: np.ndarray, scores: np.ndarray, groups: np.ndarray,
                    n_boot: int = N_BOOT, seed: int = 0,
                    multiplicities: tuple[np.ndarray, np.ndarray] | None = None,
                    chunk: int = 256, return_samples: bool = False):
    """Cluster-bootstrap CI for AUROC.

    AUROC depends on the joint ordering of all scores and has no per-scene
    decomposition, so this weights a single global sort by each resample's
    multiplicities. Chunking bounds peak memory: the full weight matrix would be
    ``n_boot x n_rows``, which is hundreds of megabytes at the corpus's size.
    """
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)
    mult, index = multiplicities or group_multiplicities(groups, n_boot, seed)

    order = np.argsort(scores, kind="mergesort")
    y_sorted = y_true[order]
    index_sorted = index[order]
    positive = (y_sorted == 1).astype(float)
    negative = 1.0 - positive

    samples = np.empty(mult.shape[0], dtype=float)
    for start in range(0, mult.shape[0], chunk):
        block = mult[start:start + chunk][:, index_sorted]          # (b, n_rows)
        weighted_neg = block * negative
        # Negatives strictly before each row, plus half the weight tied at it --
        # the midrank convention, so exact score ties do not inflate the area.
        cumulative = np.cumsum(weighted_neg, axis=1) - weighted_neg
        weighted_pos = block * positive
        wins = (weighted_pos * (cumulative + 0.5 * weighted_neg)).sum(axis=1)
        n_pos = weighted_pos.sum(axis=1)
        n_neg = weighted_neg.sum(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            samples[start:start + chunk] = np.where((n_pos > 0) & (n_neg > 0),
                                                    wins / (n_pos * n_neg), np.nan)

    result = _percentile_ci(samples, auroc(y_true, scores), "auroc")
    return (result, samples) if return_samples else result


def bootstrap_metric(metric: str, y_true: np.ndarray, y_pred: np.ndarray, groups: np.ndarray,
                     n_boot: int = N_BOOT, seed: int = 0,
                     multiplicities: tuple[np.ndarray, np.ndarray] | None = None,
                     return_samples: bool = False):
    """Dispatch to the right bootstrap for ``metric``."""
    if metric == "r2":
        return bootstrap_r2(y_true, y_pred, groups, n_boot, seed, multiplicities, return_samples)
    if metric == "balanced_accuracy":
        return bootstrap_balanced_accuracy(y_true, y_pred, groups, n_boot, seed,
                                           multiplicities, return_samples)
    if metric == "auroc":
        return bootstrap_auroc(y_true, y_pred, groups, n_boot, seed,
                               multiplicities, return_samples=return_samples)
    raise ValueError(f"unknown metric {metric!r}")


# --------------------------------------------------------------------------- #
# Multiple comparisons and equivalence
# --------------------------------------------------------------------------- #

def holm(pvals: Sequence[float]) -> np.ndarray:
    """Holm-Bonferroni adjusted p-values (spec 11).

    Uniformly more powerful than Bonferroni at the same familywise error rate.
    The family must be stated explicitly in the paper; this function only
    adjusts whatever it is handed.
    """
    p = np.asarray(pvals, dtype=float)
    m = p.size
    if m == 0:
        return p
    order = np.argsort(p)
    adjusted = np.empty(m, dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * float(p[idx]))
        adjusted[idx] = min(1.0, running)
    return adjusted


def tost_bootstrap(difference_samples: np.ndarray, margin: float,
                   alpha: float = 0.05, observed: float | None = None) -> dict:
    """Two one-sided tests for equivalence, from a paired bootstrap (spec 11).

    H2 claims mass and friction are *not* encoded. A non-significant difference
    is not evidence of absence -- spec 11 names reporting "p > 0.05, therefore
    absent" as the single most likely way this paper is rejected. Equivalence
    testing inverts the burden: it asks whether the difference from the
    random-init baseline is small enough to *exclude* an effect as large as
    ``margin``.

    ``difference_samples`` must come from a bootstrap that resampled the same
    scenes for both models (pass one shared multiplicity matrix), so the
    difference is paired.

    Equivalence is declared when the central ``1 - 2*alpha`` interval lies
    entirely inside ``(-margin, +margin)``, which is the standard TOST
    formulation and is what the two one-sided p-values below also encode.
    """
    samples = np.asarray(difference_samples, dtype=float)
    samples = samples[np.isfinite(samples)]
    if samples.size == 0:
        return {"equivalent": False, "reason": "no finite bootstrap samples"}

    point = float(observed if observed is not None else np.mean(samples))
    low, high = np.percentile(samples, [100 * alpha, 100 * (1 - alpha)])

    # One-sided bootstrap p-values: the share of resamples failing each bound.
    p_upper = float(np.mean(samples >= margin))    # H0: difference >= +margin
    p_lower = float(np.mean(samples <= -margin))   # H0: difference <= -margin

    return {
        "equivalent": bool(low > -margin and high < margin),
        "observed_difference": point,
        "margin": float(margin),
        "alpha": float(alpha),
        "ci_low": float(low),
        "ci_high": float(high),
        "p_upper": p_upper,
        "p_lower": p_lower,
        "p_tost": max(p_upper, p_lower),
        "n_boot": int(samples.size),
    }
