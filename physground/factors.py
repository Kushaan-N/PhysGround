"""Factor definitions, per-scene sampling, and the decorrelation gate (spec 5.3, 9.1).

The whole experiment rests on one property: no physical factor is predictable
from any appearance factor. If mass can be read off object size, a "mass probe"
is a size probe and every number downstream is void (spec 6.1). This module is
where that property is *created* (independent sampling) and where it is
*asserted* (:func:`check_decorrelation`).

Two deliberate departures from the spec are implemented here. Both are recorded
in DEVIATIONS.md; the reasoning is inline at the point of departure.
"""

from __future__ import annotations

import colorsys
import hashlib
from dataclasses import dataclass, field
from typing import Callable, Literal, Sequence

import numpy as np

__all__ = [
    "FactorSpec",
    "FACTORS",
    "PHYSICAL",
    "LAYOUT",
    "APPEARANCE",
    "GEOM_TYPES",
    "scene_id",
    "scene_rng",
    "sample_factors",
    "factors_table",
    "correlation_frame",
    "spearman_matrix",
    "check_decorrelation",
    "hue_to_rgb",
]

Group = Literal["physical", "layout", "appearance"]

GEOM_TYPES = ("box", "sphere", "cylinder")


# --------------------------------------------------------------------------- #
# Factor table
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class FactorSpec:
    """One independently sampled scene factor.

    ``sampler`` receives a private ``Generator`` (see :func:`scene_rng`) and
    returns a scalar. It must consume the generator deterministically and must
    not read any other factor: independence across factors is what the whole
    design buys, and a sampler that peeked at another value would destroy it
    silently.
    """

    name: str
    group: Group
    kind: Literal["continuous", "categorical", "binary"]
    sampler: Callable[[np.random.Generator], object]
    doc: str = ""
    categories: tuple[str, ...] = field(default=())


def _loguniform(lo: float, hi: float) -> Callable[[np.random.Generator], float]:
    log_lo, log_hi = np.log(lo), np.log(hi)
    return lambda rng: float(np.exp(rng.uniform(log_lo, log_hi)))


def _uniform(lo: float, hi: float) -> Callable[[np.random.Generator], float]:
    return lambda rng: float(rng.uniform(lo, hi))


def _bernoulli(p: float) -> Callable[[np.random.Generator], int]:
    return lambda rng: int(rng.random() < p)


def _choice(options: Sequence[str]) -> Callable[[np.random.Generator], str]:
    return lambda rng: str(options[int(rng.integers(len(options)))])


def _spawn_height(rng: np.random.Generator) -> float:
    """Height above resting pose at which the target is spawned.

    Not in the spec's factor table. It exists to keep ``support_state`` from
    being constant, which spec 6.3 and 14 both name as a failure mode that
    "returns a meaningless number that looks like a result". A scripted planar
    push never lifts the object, so without a drop every frame in the corpus has
    support_state=1 and the target must be discarded outright.

    A point mass at zero keeps a substantial fraction of scenes starting flat on
    the floor, so the class is defined by an actual physical difference rather
    than by a monotone function of one continuous nuisance.
    """
    if rng.random() < 0.35:
        return 0.0
    return float(rng.uniform(0.02, 0.20))


FACTORS: tuple[FactorSpec, ...] = (
    # ---- physical: the probe targets and what drives them (spec 5.3) --------
    FactorSpec("mass", "physical", "continuous", _loguniform(0.05, 2.0),
               "kg, log-uniform. Set explicitly on the geom; never via density (spec 6.1)."),
    FactorSpec("friction_slide", "physical", "continuous", _uniform(0.05, 1.2),
               "First component of geom friction."),
    FactorSpec("spawn_height", "physical", "continuous", _spawn_height,
               "Drop height above resting pose. Source of support_state variance."),

    # ---- layout: geometry/timing nuisances ---------------------------------
    FactorSpec("obj_radius", "layout", "continuous", _uniform(0.26, 0.36),
               "Target distance from the arm base, m."),
    FactorSpec("approach_angle", "layout", "continuous", _uniform(-0.52, 0.52),
               "Push direction, rad from +x. +/-30 deg keeps the workspace in frame."),
    FactorSpec("push_dist", "layout", "continuous", _uniform(0.08, 0.18),
               "How far past the target the end-effector travels, m."),
    FactorSpec("obj_yaw", "layout", "continuous", _uniform(0.0, np.pi / 2),
               "Target yaw at spawn, rad. Kept flat so boxes land on a face."),
    FactorSpec("distractor_angle", "layout", "continuous", _uniform(0.75, 1.35),
               "Angular offset of the distractor from the push ray, rad."),
    FactorSpec("distractor_side", "layout", "binary", _bernoulli(0.5),
               "Which side of the push ray the distractor sits on."),
    FactorSpec("distractor_radius", "layout", "continuous", _uniform(0.20, 0.40),
               "Distractor distance from the arm base, m."),

    # ---- appearance: must be decorrelated from everything above ------------
    FactorSpec("obj_size", "appearance", "continuous", _uniform(0.03, 0.09),
               "Characteristic half-extent, m. The factor mass must NOT track (spec 6.1)."),
    FactorSpec("obj_hue", "appearance", "continuous", _uniform(0.0, 1.0),
               "Hue in [0,1); saturation and value are fixed."),
    FactorSpec("obj_geom_type", "appearance", "categorical", _choice(GEOM_TYPES),
               "box | sphere | cylinder.", GEOM_TYPES),
    FactorSpec("floor_gray", "appearance", "continuous", _uniform(0.62, 0.86),
               "Floor lightness; a light-gray band."),
    FactorSpec("light_x", "appearance", "continuous", _uniform(-0.35, 0.35), "Light jitter, m."),
    FactorSpec("light_y", "appearance", "continuous", _uniform(-0.35, 0.35), "Light jitter, m."),
    FactorSpec("light_z", "appearance", "continuous", _uniform(1.60, 2.40), "Light height, m."),
    FactorSpec("cam_daz", "appearance", "continuous", _uniform(-2.0, 2.0), "Camera azimuth jitter, deg."),
    FactorSpec("cam_del", "appearance", "continuous", _uniform(-2.0, 2.0), "Camera elevation jitter, deg."),
    FactorSpec("cam_ddist", "appearance", "continuous", _uniform(-0.03, 0.03),
               "Camera distance jitter, fraction."),
    FactorSpec("distractor_present", "appearance", "binary", _bernoulli(0.5),
               "Second free body, never contacted. Stops the contact probe collapsing "
               "into an object-presence probe (spec 5.1)."),
    FactorSpec("distractor_size", "appearance", "continuous", _uniform(0.03, 0.09), "m."),
    FactorSpec("distractor_hue", "appearance", "continuous", _uniform(0.0, 1.0), "Hue in [0,1)."),
    FactorSpec("distractor_geom_type", "appearance", "categorical", _choice(GEOM_TYPES),
               "box | sphere | cylinder.", GEOM_TYPES),
)

_BY_NAME = {f.name: f for f in FACTORS}
PHYSICAL = tuple(f.name for f in FACTORS if f.group == "physical")
LAYOUT = tuple(f.name for f in FACTORS if f.group == "layout")
APPEARANCE = tuple(f.name for f in FACTORS if f.group == "appearance")


# --------------------------------------------------------------------------- #
# Deterministic sampling
# --------------------------------------------------------------------------- #

def scene_id(condition: str, index: int) -> str:
    """Spec 5.4: ``f"{condition}_{index:05d}"``."""
    return f"{condition}_{index:05d}"


def _name_key(name: str) -> int:
    """Stable 64-bit key for a factor name.

    Not Python's ``hash``. Spec 5.4 suggests seeding a per-scene generator from
    ``hash``, but ``hash`` on ``str`` is salted per interpreter by PYTHONHASHSEED,
    so it returns different values in different processes. Using it would make
    scene sampling depend on which worker ran it, violating the determinism
    requirement in spec 0.5 -- and it would do so silently, since every
    individual run looks self-consistent. BLAKE2b is stable across processes,
    machines, and Python versions.
    """
    return int.from_bytes(hashlib.blake2b(name.encode("utf-8"), digest_size=8).digest(), "little")


def scene_rng(index: int, master_seed: int, factor_name: str) -> np.random.Generator:
    """Private generator for one (scene, factor) pair.

    Every factor draws from its own stream rather than sharing one sequential
    stream per scene. Two consequences, both load-bearing:

    * Adding, removing, or reordering a factor leaves every other factor's
      values bit-identical, so a corpus that is half-generated when the factor
      table changes does not become silently inconsistent with its second half.
    * Independence is structural rather than incidental.

    Note the omission of ``condition``: the occlusion condition must reuse the
    factor values of the matching base scene so the comparison is paired
    (spec 5.4), so the stream depends on ``index`` and ``master_seed`` only.
    """
    seq = np.random.SeedSequence(entropy=int(master_seed), spawn_key=(int(index), _name_key(factor_name)))
    return np.random.default_rng(seq)


def sample_factors(index: int, master_seed: int, condition: str = "base") -> dict:
    """All factors for one scene. Pure in ``(index, master_seed)``.

    ``condition`` is recorded but never influences a sampled value; it only
    decides whether the occluder geom is added downstream.
    """
    out: dict = {
        "scene_index": int(index),
        "condition": condition,
        "scene_id": scene_id(condition, index),
        "master_seed": int(master_seed),
    }
    for spec in FACTORS:
        out[spec.name] = spec.sampler(scene_rng(index, master_seed, spec.name))
    return out


def factors_table(indices: Sequence[int], master_seed: int, condition: str = "base") -> dict[str, np.ndarray]:
    """Column-oriented factor table for many scenes.

    Cheap enough (microseconds per scene, no physics, no rendering) that the
    decorrelation gate can be evaluated at full corpus size rather than on the
    pilot -- see :func:`check_decorrelation`.
    """
    rows = [sample_factors(i, master_seed, condition) for i in indices]
    cols: dict[str, np.ndarray] = {}
    for key in rows[0]:
        values = [r[key] for r in rows]
        cols[key] = np.array(values, dtype=object if isinstance(values[0], str) else None)
    return cols


def hue_to_rgb(hue: float, sat: float = 0.75, val: float = 0.85) -> tuple[float, float, float]:
    """Hue in [0,1) to RGB at fixed saturation and value (spec 5.3)."""
    return colorsys.hsv_to_rgb(float(hue) % 1.0, sat, val)


# --------------------------------------------------------------------------- #
# Decorrelation (Gate G1)
# --------------------------------------------------------------------------- #

def correlation_frame(table: dict[str, np.ndarray]) -> tuple[np.ndarray, list[str], list[str]]:
    """Expand a factor table into a numeric matrix suitable for Spearman.

    Two expansions matter:

    * **Categoricals** become one-hot indicator columns. Spearman on an integer
      code for {box, sphere, cylinder} measures correlation with an arbitrary
      alphabetical ordering, so a genuine dependence on "is a sphere" can cancel
      against one on "is a box" and read as zero.
    * **Hue** gains sin/cos columns alongside the raw value. Hue is circular:
      a dependence concentrated near the 0/1 wrap-around is invisible to a rank
      correlation on the raw value.
    """
    columns: list[np.ndarray] = []
    names: list[str] = []
    groups: list[str] = []

    for spec in FACTORS:
        values = table[spec.name]
        if spec.kind == "categorical":
            for level in spec.categories:
                columns.append(np.asarray([v == level for v in values], dtype=float))
                names.append(f"{spec.name}={level}")
                groups.append(spec.group)
        else:
            columns.append(np.asarray(values, dtype=float))
            names.append(spec.name)
            groups.append(spec.group)
            if spec.name.endswith("hue"):
                radians = 2 * np.pi * np.asarray(values, dtype=float)
                columns.append(np.sin(radians)); names.append(f"{spec.name}_sin"); groups.append(spec.group)
                columns.append(np.cos(radians)); names.append(f"{spec.name}_cos"); groups.append(spec.group)

    return np.column_stack(columns), names, groups


def _rankdata(a: np.ndarray) -> np.ndarray:
    """Average-rank transform with ties averaged; matches ``scipy.stats.rankdata``.

    Fully vectorised. Tie handling is not optional here: the one-hot and binary
    columns consist of nothing but ties, and ranking them without averaging
    would impose an arbitrary within-group order and report correlations that
    are artifacts of ``argsort``'s tie-breaking.
    """
    n = a.shape[0]
    order = np.argsort(a, kind="mergesort")
    sorted_a = a[order]
    # Start index of each run of equal values.
    starts = np.flatnonzero(np.r_[True, sorted_a[1:] != sorted_a[:-1]])
    counts = np.diff(np.r_[starts, n])
    # Ranks are 1-based, so a run at [start, start+count) averages to
    # ((start+1) + (start+count)) / 2 == start + (count+1)/2.
    run_rank = starts + (counts + 1) / 2.0
    ranks = np.empty(n, dtype=float)
    ranks[order] = np.repeat(run_rank, counts)
    return ranks


def spearman_matrix(matrix: np.ndarray) -> np.ndarray:
    """Spearman rho between every pair of columns.

    Rank once per column then take a Pearson correlation, which is the
    definition, rather than calling a pairwise routine O(k^2) times. Constant
    columns yield rho=0 instead of NaN so a degenerate factor shows up in the
    label-sanity gate rather than poisoning the decorrelation matrix.
    """
    ranked = np.column_stack([_rankdata(matrix[:, j]) for j in range(matrix.shape[1])])
    centred = ranked - ranked.mean(axis=0, keepdims=True)
    norms = np.linalg.norm(centred, axis=0)
    constant = norms < 1e-12
    norms[constant] = 1.0
    normalised = centred / norms
    rho = normalised.T @ normalised
    rho[constant, :] = 0.0
    rho[:, constant] = 0.0
    np.fill_diagonal(rho, 1.0)
    return np.clip(rho, -1.0, 1.0)


def check_decorrelation(
    table: dict[str, np.ndarray],
    *,
    threshold: float = 0.10,
    alpha: float = 0.01,
    cross_group_only: bool = True,
) -> dict:
    """Gate G1. Returns a report; the caller decides whether to raise.

    Spec 9.1 says hard-fail if ``|rho| > 0.10`` for any pair, and spec 9 says to
    run the gates on a 50-scene pilot. Those two instructions are incompatible,
    and taking them literally would make G1 fail essentially always for reasons
    unrelated to the sampler:

        Under independence, Spearman rho has standard error ~1/sqrt(n-1). At
        n=50 that is 0.143, so a *correctly* independent pair exceeds 0.10 about
        48% of the time. Across the ~300 pairs in this factor table, a passing
        run would be a miracle. The gate as literally written measures pilot
        size, not decorrelation.

    Decorrelation is a property of the sampler, not of any particular draw, and
    sampling factors is free (no physics, no rendering). So this gate is
    evaluated at full corpus size, where SE ~0.018 and the 0.10 threshold sits
    at about 5.5 sigma -- a real bound rather than a coin flip. Callers pass a
    full-size table; ``run_gates.py`` does exactly that.

    Both criteria are reported, and a pair fails if either fires:

    * ``max_abs_rho > threshold`` -- the spec's effect-size bound.
    * Holm-corrected Fisher-z p-value < ``alpha`` -- an n-aware significance
      bound, which is what actually catches a small but systematic coupling in a
      large table.
    """
    matrix, names, groups = correlation_frame(table)
    n = matrix.shape[0]
    rho = spearman_matrix(matrix)

    iu = np.triu_indices(len(names), k=1)
    pair_rho = rho[iu]
    same_group = np.array([groups[i] == groups[j] for i, j in zip(*iu)])
    # Columns expanded from one factor (hue and its sin/cos, one-hot levels of a
    # single categorical) are correlated with each other by construction. That is
    # arithmetic, not a sampling failure, so exclude those pairs.
    same_factor = np.array([_base_factor(names[i]) == _base_factor(names[j]) for i, j in zip(*iu)])

    considered = ~same_factor
    if cross_group_only:
        considered &= ~same_group

    # Fisher z. Valid for Spearman to good approximation with the 1.06 variance
    # inflation; n<=3 has no usable test, so leave those p-values at 1.
    if n > 3:
        z = np.arctanh(np.clip(pair_rho, -0.999999, 0.999999)) * np.sqrt((n - 3) / 1.06)
        pvals = 2.0 * _norm_sf(np.abs(z))
    else:
        pvals = np.ones_like(pair_rho)

    holm = _holm(pvals[considered]) if considered.any() else np.array([])
    holm_full = np.ones_like(pvals)
    holm_full[considered] = holm

    rows = []
    for k, (i, j) in enumerate(zip(*iu)):
        rows.append({
            "a": names[i], "b": names[j],
            "group_a": groups[i], "group_b": groups[j],
            "rho": float(pair_rho[k]),
            "p_holm": float(holm_full[k]),
            "considered": bool(considered[k]),
        })

    if considered.any():
        max_abs = float(np.abs(pair_rho[considered]).max())
        worst = int(np.flatnonzero(considered)[int(np.argmax(np.abs(pair_rho[considered])))])
        worst_pair = (names[iu[0][worst]], names[iu[1][worst]])
        n_sig = int((holm_full[considered] < alpha).sum())
    else:
        max_abs, worst_pair, n_sig = 0.0, ("", ""), 0

    passed = (max_abs <= threshold) and (n_sig == 0)
    return {
        "gate": "G1",
        "passed": bool(passed),
        "n_scenes": int(n),
        "threshold": float(threshold),
        "alpha": float(alpha),
        "max_abs_rho": max_abs,
        "worst_pair": list(worst_pair),
        "n_significant_after_holm": n_sig,
        "n_pairs_considered": int(considered.sum()),
        "names": names,
        "groups": groups,
        "rho_matrix": rho,
        "pairs": rows,
    }


def _base_factor(column_name: str) -> str:
    """Map an expanded column back to the factor it came from."""
    name = column_name.split("=", 1)[0]
    for suffix in ("_sin", "_cos"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    return name


def _norm_sf(x: np.ndarray) -> np.ndarray:
    """Upper tail of the standard normal, via erfc. Avoids a scipy import here."""
    from math import erfc
    return np.array([0.5 * erfc(float(v) / np.sqrt(2.0)) for v in np.atleast_1d(x)]).reshape(np.shape(x))


def _holm(pvals: np.ndarray) -> np.ndarray:
    """Holm-Bonferroni adjusted p-values (spec 11)."""
    p = np.asarray(pvals, dtype=float)
    m = p.size
    order = np.argsort(p)
    adjusted = np.empty(m, dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        adjusted[idx] = min(1.0, running)
    return adjusted
