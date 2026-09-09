"""Experiments 0 through G (spec 10, plus H5).

Each experiment writes raw per-item predictions and targets to ``.npz`` and
nothing else (spec 0.4). Summary tables and figures are regenerated from those
arrays by ``scripts/make_figures.py``, so no number in the paper depends on
re-running a model, and a statistic can be revised without recomputing features.

Cell keys
---------
A "cell" is one (encoder, condition, layer, view, task) combination. Its arrays
are stored under ``"{cell}|{target}|y_true"`` and friends in a single archive
per (experiment, seed). Flat string keys rather than nested groups because
``.npz`` has no hierarchy, and inventing one via pickled objects would make
every results file capable of executing code on load.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from . import features as feature_module
from . import paths as P
from .probes import LogisticProbe, MultiRidgeCV, shuffle_labels_across_scenes
from .splits import scene_split, frame_mask

__all__ = [
    "cell_key",
    "run_cell",
    "experiment_a",
    "experiment_b",
    "experiment_c",
    "experiment_d",
    "experiment_e",
    "experiment_f",
    "experiment_g",
    "EXPERIMENTS",
    "MAIN_GRID_LAYERS",
]

#: Spec 7.2 for the ViT-B encoders. Video encoders carry their own layer sets.
MAIN_GRID_LAYERS = (2, 5, 8, 11)


def cell_key(encoder: str, condition: str, layer: int, view: str, task: str = "real") -> str:
    return f"{encoder}|{condition}|L{layer}|{view}|{task}"


# --------------------------------------------------------------------------- #
# One cell
# --------------------------------------------------------------------------- #

def run_cell(encoder: str, condition: str, layer: int, view: str, seed: int,
             targets: Sequence[str], *, root: Path | None = None, control: bool = False,
             pool_scenes: bool = False, eval_conditions: Sequence[str] = (),
             n_folds: int = 5, train_frac: float = 0.8) -> dict[str, np.ndarray]:
    """Fit probes for one cell and return raw held-out predictions.

    ``control`` runs the selectivity task of spec 8.5: identical features and
    identical split, with labels permuted between scenes. It must land at
    chance; if it does not, the features are leaking scene identity.

    ``pool_scenes`` averages a scene's frames into one row. This exists so the
    frame-versus-video comparison of H3 is matched: video encoders emit one
    embedding per scene (spec 12), and comparing a per-scene R2 against a
    per-frame R2 would confound the encoder with the number of observations it
    was scored on.

    ``eval_conditions`` evaluates the fitted probe on further conditions without
    refitting -- the H4 setting, where a probe trained on clean frames meets
    occlusion at test time, exactly as a latent world model trained on clean
    data would.
    """
    x, target_table, scene_index = feature_module.load_dataset(encoder, condition, layer, view, root)

    if pool_scenes and np.unique(scene_index).size != scene_index.size:
        x, target_table, scene_index = _pool_by_scene(x, target_table, scene_index)

    train_scenes, test_scenes = scene_split(scene_index, seed, train_frac)
    train = frame_mask(scene_index, train_scenes)
    test = frame_mask(scene_index, test_scenes)

    available = [t for t in targets if t in target_table]
    regression = [t for t in available if t in feature_module.REGRESSION_TARGETS]
    classification = [t for t in available if t in feature_module.CLASSIFICATION_TARGETS]

    labels = {t: np.asarray(target_table[t], dtype=float) for t in available}
    if control:
        labels = {t: shuffle_labels_across_scenes(v, scene_index, seed + 9973)
                  for t, v in labels.items()}

    out: dict[str, np.ndarray] = {}
    evaluations = [(condition, x, labels, scene_index, test)]
    for other in eval_conditions:
        evaluations.append(_matched_evaluation(encoder, other, layer, view, root,
                                               test_scenes, available, control, seed, pool_scenes))

    if regression:
        y_train = np.column_stack([labels[t][train] for t in regression])
        model = MultiRidgeCV(n_folds=n_folds, seed=seed).fit(x[train], y_train, scene_index[train])
        for name, x_eval, label_eval, scenes_eval, mask in evaluations:
            prediction = model.predict(x_eval[mask])
            for k, t in enumerate(regression):
                prefix = f"{name}|{t}"
                out[f"{prefix}|y_true"] = label_eval[t][mask]
                out[f"{prefix}|y_pred"] = prediction[:, k]
                out[f"{prefix}|scene_index"] = scenes_eval[mask]
        out["ridge_alpha"] = np.asarray(model.best_alpha_, dtype=float)
        out["ridge_targets"] = np.asarray(regression)

    for t in classification:
        y = labels[t].astype(int)
        if np.unique(y[train]).size < 2:
            continue
        probe = LogisticProbe(n_folds=n_folds, seed=seed).fit(x[train], y[train], scene_index[train])
        for name, x_eval, label_eval, scenes_eval, mask in evaluations:
            prefix = f"{name}|{t}"
            out[f"{prefix}|y_true"] = label_eval[t][mask].astype(int)
            out[f"{prefix}|y_pred"] = probe.predict(x_eval[mask])
            out[f"{prefix}|y_score"] = probe.decision_function(x_eval[mask])
            out[f"{prefix}|scene_index"] = scenes_eval[mask]
        out[f"logistic_C|{t}"] = np.asarray([probe.best_c_], dtype=float)

    # The occlusion covariate travels with the predictions so the H4 figure can
    # bin by it without reopening the corpus.
    for name, _, _, _, mask in evaluations:
        covariate = _occlusion_for(name, mask, root, pool_scenes)
        if covariate is not None:
            out[f"{name}|occlusion_fraction"] = covariate
    return out


def _pool_by_scene(x, target_table, scene_index):
    """Average each scene's frames into one row.

    Per-frame targets are collapsed to the scene's first frame, which is only
    meaningful for targets constant within a scene -- callers restrict to
    ``PER_SCENE_TARGETS``.
    """
    scenes, inverse = np.unique(scene_index, return_inverse=True)
    pooled = np.zeros((scenes.size, x.shape[1]), dtype=x.dtype)
    counts = np.bincount(inverse, minlength=scenes.size).astype(float)
    np.add.at(pooled, inverse, x)
    pooled /= counts[:, None]
    first = np.zeros(scenes.size, dtype=np.int64)
    first[inverse[::-1]] = np.arange(scene_index.size)[::-1]
    collapsed = {key: np.asarray(values)[first] for key, values in target_table.items()}
    return pooled, collapsed, scenes


def _matched_evaluation(encoder, condition, layer, view, root, test_scenes, targets,
                        control, seed, pool_scenes):
    """Load another condition restricted to the same held-out scenes.

    Matched pairs share every sampled factor and the whole scripted trajectory
    (spec 5.4), so restricting to the same test scenes makes the comparison
    paired: the only difference between the two evaluations is the occluder.
    """
    x, table, scene_index = feature_module.load_dataset(encoder, condition, layer, view, root)
    if pool_scenes and np.unique(scene_index).size != scene_index.size:
        x, table, scene_index = _pool_by_scene(x, table, scene_index)
    labels = {t: np.asarray(table[t], dtype=float) for t in targets if t in table}
    if control:
        labels = {t: shuffle_labels_across_scenes(v, scene_index, seed + 9973)
                  for t, v in labels.items()}
    mask = frame_mask(scene_index, test_scenes)
    return condition, x, labels, scene_index, mask


def _occlusion_for(condition, mask, root, pool_scenes):
    """Per-row occlusion fraction for the evaluated rows of one condition.

    Indexed by the evaluation *mask*, not by its scene ids. Both the target
    table and the feature rows are sorted by ``(scene_index, frame_index)``, so
    the mask addresses the same rows in both; matching on scene ids instead
    would silently drop the covariate whenever a scene contributes more than one
    row, which is every non-pooled case.
    """
    try:
        table = feature_module.load_targets(condition, root=root)
    except (FileNotFoundError, OSError):
        return None
    if "occlusion_fraction" not in table:
        return None
    if pool_scenes:
        first = np.flatnonzero(np.r_[True, table["scene_index"][1:] != table["scene_index"][:-1]])
        table = {k: v[first] for k, v in table.items()}
    values = np.asarray(table["occlusion_fraction"], dtype=float)
    if values.size != np.asarray(mask).size:
        return None
    return values[mask]


# --------------------------------------------------------------------------- #
# Experiments
# --------------------------------------------------------------------------- #

def _save(exp: str, seed: int, arrays: dict[str, np.ndarray], root: Path | None = None) -> Path:
    directory = (Path(root) / "results" / exp / f"seed_{seed:02d}") if root else P.results_dir(exp, seed)
    path = directory / "raw.npz"
    P.atomic_write_npz(path, arrays, compress=True)
    P.write_done_marker(path, cfg_hash=P.config_hash({"exp": exp, "seed": seed,
                                                      "keys": sorted(arrays)}))
    return path


def _prefixed(cell: str, arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {f"{cell}|{key}": value for key, value in arrays.items()}


def experiment_a(seed: int = 0, root: Path | None = None, condition: str = "base",
                 encoder: str = "dinov2_b", layer: int = 8, view: str = "cls") -> dict:
    """Positive control: object image-plane position (spec 10, Exp A).

    Kill condition is R2 < 0.90. Object position is unambiguously present in
    the image, so failure here indicates a broken pipeline, not a finding
    (spec 6.4) -- and every null result later in the paper depends on the
    pipeline being able to detect a property that *is* there.
    """
    from .stats import r2_score

    arrays = run_cell(encoder, condition, layer, view, seed,
                      ("obj_pos_x", "obj_pos_y"), root=root)
    cell = cell_key(encoder, condition, layer, view)
    path = _save("A", seed, _prefixed(cell, arrays), root)

    scores = {t: r2_score(arrays[f"{condition}|{t}|y_true"], arrays[f"{condition}|{t}|y_pred"])
              for t in ("obj_pos_x", "obj_pos_y")}
    mean_r2 = float(np.mean(list(scores.values())))
    return {"exp": "A", "seed": seed, "path": str(path), "r2": scores, "mean_r2": mean_r2,
            "passed": bool(mean_r2 >= 0.90),
            "kill_condition": "mean R2 < 0.90 means the pipeline is broken; stop (spec 10)"}


def experiment_b(seed: int = 0, root: Path | None = None, condition: str = "base",
                 encoder: str = "dinov2_b", layer: int = 8, view: str = "cls") -> dict:
    """Selectivity control on every target (spec 10, Exp B / 8.5).

    Kill condition is a control above chance, which means split leakage.
    """
    from .stats import balanced_accuracy, r2_score

    arrays = run_cell(encoder, condition, layer, view, seed, feature_module.TARGETS,
                      root=root, control=True)
    cell = cell_key(encoder, condition, layer, view, task="control")
    path = _save("B", seed, _prefixed(cell, arrays), root)

    scores, flagged = {}, []
    for target in feature_module.TARGETS:
        true_key = f"{condition}|{target}|y_true"
        if true_key not in arrays:
            continue
        y_true, y_pred = arrays[true_key], arrays[f"{condition}|{target}|y_pred"]
        if target in feature_module.CLASSIFICATION_TARGETS:
            value = balanced_accuracy(y_true, y_pred)
            over = value > 0.60
        else:
            value = r2_score(y_true, y_pred)
            over = value > 0.10
        scores[target] = float(value)
        if over:
            flagged.append(target)

    return {"exp": "B", "seed": seed, "path": str(path), "control_scores": scores,
            "above_chance": flagged, "passed": not flagged,
            "kill_condition": "a control above chance means scene identity is leaking "
                              "across the split (spec 7.4, 8.5)"}


def experiment_c(seed: int = 0, root: Path | None = None, condition: str = "base",
                 encoders: Sequence[str] = ("random_b", "raw_pixel")) -> dict:
    """Baselines across all targets (spec 10, Exp C / 8.4)."""
    arrays: dict[str, np.ndarray] = {}
    cells = []
    for encoder in encoders:
        layers = (0,) if encoder == "raw_pixel" else MAIN_GRID_LAYERS
        views = ("mean",) if encoder == "raw_pixel" else ("cls", "mean")
        for layer in layers:
            for view in views:
                cell = cell_key(encoder, condition, layer, view)
                arrays.update(_prefixed(cell, run_cell(
                    encoder, condition, layer, view, seed, feature_module.TARGETS, root=root)))
                cells.append(cell)
    path = _save("C", seed, arrays, root)
    return {"exp": "C", "seed": seed, "path": str(path), "n_cells": len(cells)}


def experiment_d(seed: int = 0, root: Path | None = None, condition: str = "base",
                 encoders: Sequence[str] = ("dinov2_b", "random_b"),
                 layers: Sequence[int] = MAIN_GRID_LAYERS,
                 views: Sequence[str] = ("cls", "mean")) -> dict:
    """The main grid (spec 10, Exp D).

    2 encoders x 4 layers x 2 views x every target. All targets in a cell share
    one eigendecomposition, so the grid costs far less than the per-cell count
    suggests (see :mod:`physground.probes`).
    """
    arrays: dict[str, np.ndarray] = {}
    cells = []
    for encoder in encoders:
        for layer in layers:
            for view in views:
                cell = cell_key(encoder, condition, layer, view)
                arrays.update(_prefixed(cell, run_cell(
                    encoder, condition, layer, view, seed, feature_module.TARGETS, root=root)))
                cells.append(cell)
    path = _save("D", seed, arrays, root)
    return {"exp": "D", "seed": seed, "path": str(path), "n_cells": len(cells), "cells": cells}


def experiment_e(seed: int = 0, root: Path | None = None,
                 encoders: Sequence[str] = ("dinov2_b",),
                 layers: Sequence[int] = (8, 11), views: Sequence[str] = ("cls", "mean")) -> dict:
    """Occlusion, matched pairs (spec 10, Exp E / H4).

    Probes are fitted on *unoccluded* training scenes and evaluated on both the
    unoccluded and the occluded halves of the same held-out scenes. That is the
    situation H4 is about: a readout learned from clean observation meeting the
    occlusion that manipulation constantly produces. Fitting separately on each
    condition would answer a different and less interesting question -- whether
    occluded frames are learnable at all.
    """
    arrays: dict[str, np.ndarray] = {}
    cells = []
    for encoder in encoders:
        for layer in layers:
            for view in views:
                cell = cell_key(encoder, "paired", layer, view)
                arrays.update(_prefixed(cell, run_cell(
                    encoder, "base", layer, view, seed, feature_module.TARGETS,
                    root=root, eval_conditions=("occluded",))))
                cells.append(cell)
    path = _save("E", seed, arrays, root)
    return {"exp": "E", "seed": seed, "path": str(path), "n_cells": len(cells)}


def experiment_f(seed: int = 0, root: Path | None = None, condition: str = "base",
                 video_encoders: Sequence[str] = ("videomae_b", "vjepa2"),
                 frame_encoders: Sequence[str] = ("dinov2_b", "random_b")) -> dict:
    """Frame versus video encoders on the dynamic properties (spec 10, Exp F / H3).

    Restricted to :data:`PER_SCENE_TARGETS`. A video encoder returns one
    embedding per scene, so per-frame targets are not defined for it (spec 12),
    and this asymmetry means H3 is tested on dynamic properties only.

    The frame encoders are re-run here with ``pool_scenes=True`` rather than
    reusing their Experiment D numbers. Otherwise the comparison would score the
    frame encoders on ten times as many rows as the video encoders, and the
    difference in effective sample size would be indistinguishable from a
    difference in representation.
    """
    from .encoders import get_encoder  # noqa: F401  (registry side effects)

    arrays: dict[str, np.ndarray] = {}
    cells: list[str] = []

    for encoder in frame_encoders:
        for layer in MAIN_GRID_LAYERS:
            for view in ("cls", "mean"):
                cell = cell_key(encoder, condition, layer, view, task="scene_pooled")
                arrays.update(_prefixed(cell, run_cell(
                    encoder, condition, layer, view, seed, feature_module.PER_SCENE_TARGETS,
                    root=root, pool_scenes=True)))
                cells.append(cell)

    for encoder in video_encoders:
        layers = _video_layers(encoder)
        for layer in layers:
            cell = cell_key(encoder, condition, layer, "mean")
            try:
                arrays.update(_prefixed(cell, run_cell(
                    encoder, condition, layer, "mean", seed,
                    feature_module.PER_SCENE_TARGETS, root=root)))
                cells.append(cell)
            except FileNotFoundError:
                continue

    path = _save("F", seed, arrays, root)
    return {"exp": "F", "seed": seed, "path": str(path), "n_cells": len(cells), "cells": cells}


def _video_layers(encoder: str) -> tuple[int, ...]:
    if encoder == "vjepa2":
        from .encoders.vjepa2 import DEFAULT_LAYERS
        return DEFAULT_LAYERS
    from .encoders.videomae import DEFAULT_LAYERS
    return DEFAULT_LAYERS


def experiment_g(seed: int = 0, root: Path | None = None, condition: str = "base",
                 encoders: Sequence[str] = ("dinov2_b", "random_b"),
                 layers: Sequence[int] = MAIN_GRID_LAYERS,
                 views: Sequence[str] = ("cls", "mean"), epochs: int = 60) -> dict:
    """H5, optional: does probe decodability predict latent-dynamics error?

    Spec 2 marks this a stretch goal to run only after 0-F complete cleanly, and
    predicts the outcome as "unknown" -- so unlike the rest of the sequence there
    is no result here that would count as confirmation or refutation.

    For every cell already in the main grid, trains a small next-latent
    predictor on the same features and the same scene split, then reports its
    held-out R2 against the stationary baseline alongside that cell's
    contact-state probe accuracy. The correlation between those two columns is
    the quantity H5 is about; it is computed in ``summarize``, not here, so the
    raw per-cell numbers stay available if the summary statistic changes.
    """
    from .latent_dynamics import build_transitions, fit_latent_predictor
    from .probes import LogisticProbe
    from .stats import balanced_accuracy

    rows: list[dict] = []
    for encoder in encoders:
        for layer in layers:
            for view in views:
                x, targets, scene_index = feature_module.load_dataset(
                    encoder, condition, layer, view, root)
                train_scenes, test_scenes = scene_split(scene_index, seed)

                transitions = build_transitions(x, targets, scene_index)
                train = frame_mask(transitions.scene_index, train_scenes)
                test = frame_mask(transitions.scene_index, test_scenes)
                dynamics = fit_latent_predictor(transitions, train, test,
                                                epochs=epochs, seed=seed)

                frame_train = frame_mask(scene_index, train_scenes)
                frame_test = frame_mask(scene_index, test_scenes)
                contact = np.asarray(targets["contact_state"]).astype(int)
                probe = LogisticProbe(seed=seed).fit(x[frame_train], contact[frame_train],
                                                     scene_index[frame_train])
                accuracy = balanced_accuracy(contact[frame_test], probe.predict(x[frame_test]))

                rows.append({"encoder": encoder, "layer": int(layer), "view": view,
                             "contact_balanced_accuracy": float(accuracy), **dynamics})

    arrays: dict[str, np.ndarray] = {}
    for column in ("contact_balanced_accuracy", "r2_vs_stationary", "model_mse",
                   "stationary_mse", "layer", "latent_dim"):
        arrays[column] = np.array([r[column] for r in rows], dtype=float)
    arrays["cell"] = np.array([f"{r['encoder']}|L{r['layer']}|{r['view']}" for r in rows])

    path = _save("G", seed, arrays, root)
    finite = np.isfinite(arrays["contact_balanced_accuracy"]) & np.isfinite(arrays["r2_vs_stationary"])
    correlation = (float(np.corrcoef(arrays["contact_balanced_accuracy"][finite],
                                     arrays["r2_vs_stationary"][finite])[0, 1])
                   if finite.sum() > 2 else float("nan"))
    return {"exp": "G", "seed": seed, "path": str(path), "n_cells": len(rows),
            "pearson_contact_vs_dynamics_r2": correlation,
            "note": "spec 2 predicts this outcome as unknown; exploratory"}


EXPERIMENTS = {
    "A": experiment_a,
    "B": experiment_b,
    "C": experiment_c,
    "D": experiment_d,
    "E": experiment_e,
    "F": experiment_f,
    "G": experiment_g,
}
