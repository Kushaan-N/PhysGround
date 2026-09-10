"""The four validation gates (spec 9).

Spec 0.3: validation code is written before generation code, and all four gates
must pass on the pilot before the full corpus is generated. The ordering is not
negotiable because each gate is cheaper than everything it protects.

G1 runs at full corpus size rather than pilot size. Decorrelation is a property
of the *sampler*, and sampling factors costs microseconds per scene with no
physics and no rendering, so there is no reason to measure it on 50 draws where
the sampling noise swamps the effect being bounded. See
:func:`physground.factors.check_decorrelation` for the arithmetic.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from . import paths as P
from .factors import APPEARANCE, FACTORS, LAYOUT, PHYSICAL, check_decorrelation, factors_table

__all__ = ["gate_g1", "gate_g2", "gate_g3", "gate_g4", "GateFailure"]


class GateFailure(RuntimeError):
    """Raised when a gate fails and the caller asked for a hard stop."""


# --------------------------------------------------------------------------- #
# G1 - decorrelation
# --------------------------------------------------------------------------- #

#: Fewest scenes at which the |rho| <= 0.10 bound is a real bound rather than a
#: coin flip. Spearman's standard error under independence is ~1/sqrt(n-1), so
#: the threshold sits at 3.2 sigma here and at 0.8 sigma for a 50-scene pilot.
G1_MIN_SCENES = 1000


def gate_g1(n_scenes: int, master_seed: int, out_dir: Path | None = None,
            threshold: float = 0.10, alpha: float = 0.01,
            derived: dict[str, np.ndarray] | None = None,
            scene_index: np.ndarray | None = None,
            min_scenes: int = G1_MIN_SCENES) -> dict:
    """Decorrelation over the factor sampler, plus an advisory derived table.

    Two tables, and only the first is a gate.

    *Sampled factors* -- the hard gate, evaluated at the full corpus size. This
    is where the property the paper depends on actually lives: no sampled
    physical factor may be predictable from any appearance factor. It is the
    table the appendix figure shows.

    *Derived targets* (contact rate, object position, speed, occlusion) --
    **advisory only**. These exist after simulation, and applying the same
    hard-fail to them would be a category error, because most of their
    dependence on other factors is definitional rather than a leak. Measured on
    a 1500-scene corpus:

        obj_pos_y     vs approach_angle  rho = -0.950
        obj_speed     vs spawn_height    rho = +0.852
        ee_obj_dist   vs obj_size        rho = +0.486
        occlusion     vs geom type       rho = -0.707 (sphere)

    Every one of those is the quantity's own definition. The object's image
    position *is* where the object is; a dropped object *is* moving at frame 0,
    which is the entire reason spawn_height exists; the finger stops at the
    object's surface, so end-effector distance carries the object's radius; a
    sphere presents less area behind a fixed slab than a box. Failing a gate on
    these would demand that the corpus contradict its own geometry.

    The one dependence worth reading carefully is ``obj_speed`` vs
    ``friction_slide`` (rho = -0.199): higher friction really does stop the
    object sooner. That is physics, and it is precisely the signal H3 predicts a
    video encoder can exploit -- so it is a property of the corpus being correct,
    not of it being contaminated.

    What must not appear is appearance predicting *mass* or *friction*, and that
    is exactly what the sampled-factor table gates.

    Frames are aggregated to one value per scene before correlating: frames
    within a scene are not independent, so a frame-level correlation would have
    an effective sample size near the scene count while being tested as though
    it had ten times that.
    """
    # Sampling factors costs microseconds per scene, no physics and no
    # rendering, so there is never a reason to evaluate the sampler on fewer
    # draws than make the bound meaningful. Raising it silently is right here
    # precisely because the quantity does not depend on the corpus -- a caller
    # who passed a pilot size wanted the gate, not a noise measurement.
    effective = max(int(n_scenes), int(min_scenes))
    if effective != n_scenes:
        print(f"[G1] evaluating the sampler at {effective} scenes rather than {n_scenes}: "
              f"below {min_scenes} the |rho| <= {threshold} bound is within sampling noise "
              f"(SE ~ {1 / max(n_scenes - 1, 1) ** 0.5:.3f}). Decorrelation is a property of "
              "the sampler, not of the corpus, and sampling is free.", flush=True)

    table = factors_table(range(effective), master_seed)
    report = check_decorrelation(table, threshold=threshold, alpha=alpha)
    report["requested_scenes"] = int(n_scenes)

    if derived and scene_index is not None:
        advisory = _derived_decorrelation(derived, scene_index, master_seed, threshold, alpha)
        advisory["advisory"] = True
        advisory["note"] = ("diagnostic only; derived targets depend on layout and geometry "
                            "by definition. The gate is the sampled-factor table.")
        report["derived"] = advisory

    if out_dir is not None:
        out_dir = Path(out_dir)
        _write_correlation_csv(report, out_dir / "decorrelation.csv")
        summary = {k: v for k, v in report.items() if k not in ("rho_matrix", "pairs")}
        P.atomic_write_json(out_dir / "gate_g1.json", summary)
    return report


def _derived_decorrelation(derived: dict[str, np.ndarray], scene_index: np.ndarray,
                           master_seed: int, threshold: float, alpha: float) -> dict:
    scene_index = np.asarray(scene_index)
    scenes = np.unique(scene_index)
    table = factors_table(scenes.tolist(), master_seed)

    for name, values in derived.items():
        values = np.asarray(values, dtype=float)
        table[name] = np.array([values[scene_index == s].mean() for s in scenes])

    from .factors import FactorSpec
    extra = tuple(FactorSpec(name, "physical", "continuous", lambda rng: 0.0, "derived per-scene mean")
                  for name in derived)

    import physground.factors as factors_module
    original = factors_module.FACTORS
    try:
        factors_module.FACTORS = original + extra
        report = check_decorrelation(table, threshold=threshold, alpha=alpha)
    finally:
        factors_module.FACTORS = original

    return {k: v for k, v in report.items() if k not in ("rho_matrix", "pairs")}


def _write_correlation_csv(report: dict, path: Path) -> None:
    lines = ["a,b,group_a,group_b,rho,p_holm,considered"]
    for row in report["pairs"]:
        lines.append(f"{row['a']},{row['b']},{row['group_a']},{row['group_b']},"
                     f"{row['rho']:.6f},{row['p_holm']:.6g},{int(row['considered'])}")
    P.atomic_write_bytes(Path(path), "\n".join(lines).encode("utf-8"))


# --------------------------------------------------------------------------- #
# G2 - visual inspection
# --------------------------------------------------------------------------- #

def gate_g2(condition_scenes: Sequence[tuple[str, int]], out_path: Path,
            n_frames: int = 20, seed: int = 0) -> dict:
    """Render a contact sheet for a human to look at (spec 9.2).

    Deliberately returns ``passed: None``. This gate has no automated verdict --
    spec 17.6 is explicit that it "must reach your eyes" and cannot be satisfied
    by a check. Returning False would be wrong (nothing failed) and returning
    True would let a pipeline claim a human looked at something nobody opened.
    """
    from .figures import contact_sheet
    from .frames import unpack_frames

    rng = np.random.default_rng(seed)
    frames, captions = [], []
    for condition, index in condition_scenes:
        directory = P.scene_dir(condition, index)
        if not (directory / "frames.npz").exists():
            continue
        with np.load(directory / "frames.npz") as archive:
            scene_frames = unpack_frames(archive)
        with np.load(directory / "gt.npz") as gt:
            for k in rng.choice(len(scene_frames), size=min(2, len(scene_frames)), replace=False):
                frames.append(scene_frames[k])
                captions.append(
                    f"{condition[:3]}{index:05d} f{k} {str(gt['phase'][k])[:16]}\n"
                    f"contact={gt['contact_state'][k]} support={gt['support_state'][k]} "
                    f"occ={gt['occlusion_fraction'][k]:.2f}\n"
                    f"speed={gt['obj_speed'][k]:.2f} mass={gt['mass'][k]:.2f} "
                    f"fric={gt['friction_slide'][k]:.2f}")

    if not frames:
        raise GateFailure("G2: no frames found to build a contact sheet")

    order = rng.permutation(len(frames))[:n_frames]
    path = contact_sheet([frames[i] for i in order], [captions[i] for i in order],
                         Path(out_path), ncols=5,
                         title="G2 contact sheet - open this and look at it (spec 9.2)")
    return {"gate": "G2", "passed": None, "path": str(path), "n_frames": len(order),
            "note": "requires human inspection; no automated verdict"}


# --------------------------------------------------------------------------- #
# G3 - label sanity
# --------------------------------------------------------------------------- #

def gate_g3(targets: dict[str, np.ndarray], scene_index: np.ndarray,
            out_dir: Path | None = None) -> dict:
    """Label sanity (spec 9.3)."""
    from scipy import stats as scipy_stats

    scene_index = np.asarray(scene_index)
    checks: list[dict] = []

    contact = np.asarray(targets["contact_state"], dtype=float)
    rate = float(contact.mean())
    checks.append({"name": "contact_positive_rate", "value": rate,
                   "passed": bool(0.25 <= rate <= 0.55), "bounds": [0.25, 0.55]})

    support = np.asarray(targets["support_state"], dtype=float)
    support_rate = float(support.mean())
    checks.append({"name": "support_not_constant", "value": support_rate,
                   "passed": bool(0.0 < support_rate < 1.0),
                   "note": "constant support_state must be dropped from the probe set (spec 6.3)"})

    for name, values in targets.items():
        array = np.asarray(values)
        if array.dtype.kind not in "fiu":
            continue
        bad = int(np.sum(~np.isfinite(array.astype(float))))
        if bad:
            checks.append({"name": f"finite:{name}", "value": bad, "passed": False})
    if not any(c["name"].startswith("finite:") for c in checks):
        checks.append({"name": "all_targets_finite", "value": 0, "passed": True})

    # KS tests use one value per scene. mass and friction are constant within a
    # scene, so testing per-frame would repeat each draw ten times: the empirical
    # CDF is unchanged but n inflates tenfold, and the test rejects a correct
    # sampler on trivial deviations.
    first = np.flatnonzero(np.r_[True, scene_index[1:] != scene_index[:-1]])
    for name, cdf in (("mass", _loguniform_cdf(0.05, 2.0)),
                      ("friction_slide", _uniform_cdf(0.05, 1.2))):
        if name not in targets:
            continue
        per_scene = np.asarray(targets[name], dtype=float)[first]
        result = scipy_stats.kstest(per_scene, cdf)
        checks.append({"name": f"ks:{name}", "value": float(result.pvalue),
                       "statistic": float(result.statistic), "n": int(per_scene.size),
                       "passed": bool(result.pvalue > 0.01), "bounds": [0.01, 1.0]})

    report = {"gate": "G3", "passed": all(c["passed"] for c in checks),
              "n_frames": int(scene_index.size), "n_scenes": int(np.unique(scene_index).size),
              "checks": checks}
    if out_dir is not None:
        P.atomic_write_json(Path(out_dir) / "gate_g3.json", report)
    return report


def _loguniform_cdf(low: float, high: float):
    log_low, log_high = np.log(low), np.log(high)
    return lambda x: np.clip((np.log(np.clip(x, 1e-12, None)) - log_low) / (log_high - log_low), 0, 1)


def _uniform_cdf(low: float, high: float):
    return lambda x: np.clip((x - low) / (high - low), 0, 1)


# --------------------------------------------------------------------------- #
# G4 - precision
# --------------------------------------------------------------------------- #

def gate_g4(condition: str = "base", encoder: str = "dinov2_b", layer: int = 8,
            view: str = "cls", tolerance: float = 0.01, probe_seed: int = 0,
            out_dir: Path | None = None, root: Path | None = None) -> dict:
    """fp16 versus fp32 on the positive control (spec 9.4).

    Extracts the pilot twice and compares R2 on object position. Once this
    passes, fp16 is used everywhere and the question is closed. The comparison
    is made on the *probe metric*, not on feature reconstruction error, because
    that is the only quantity any conclusion depends on -- features can differ
    at the fifth decimal without moving a linear readout at all.
    """
    from . import features as feature_module
    from .probes import MultiRidgeCV
    from .splits import split_frames
    from .stats import r2_score

    scores = {}
    for dtype in ("float32", "float16"):
        for shard in range(1):
            feature_module.extract(encoder, condition, shard, 1, root=root, dtype=dtype,
                                   store_patches=False, force=True, progress_every=0)
        x, targets, scene_index = feature_module.load_dataset(encoder, condition, layer, view, root)
        train, test = split_frames(scene_index, probe_seed)
        y = np.column_stack([targets["obj_pos_x"], targets["obj_pos_y"]]).astype(float)
        model = MultiRidgeCV().fit(x[train], y[train], scene_index[train])
        prediction = model.predict(x[test])
        scores[dtype] = float(np.mean([r2_score(y[test][:, k], prediction[:, k]) for k in range(2)]))

    difference = abs(scores["float32"] - scores["float16"])
    report = {"gate": "G4", "passed": bool(difference < tolerance),
              "r2_float32": scores["float32"], "r2_float16": scores["float16"],
              "abs_difference": difference, "tolerance": tolerance,
              "encoder": encoder, "layer": layer, "view": view}
    if out_dir is not None:
        P.atomic_write_json(Path(out_dir) / "gate_g4.json", report)
    return report
