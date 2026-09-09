"""Probes, splits, and statistics (spec 3, 8, 11)."""

from __future__ import annotations

import numpy as np
import pytest

from physground import splits as SP
from physground import stats as ST
from physground.probes import (LogisticProbe, MultiRidgeCV, RIDGE_ALPHAS,
                               shuffle_labels_across_scenes)


@pytest.fixture(scope="module")
def dataset():
    rng = np.random.default_rng(0)
    n_scenes, per_scene, dim = 200, 10, 48
    scene_index = np.repeat(np.arange(n_scenes), per_scene)
    x = rng.normal(size=(scene_index.size, dim)) * rng.uniform(0.5, 3, dim) + rng.uniform(-2, 2, dim)
    weights = rng.normal(size=(dim, 2))
    y = x @ weights + rng.normal(size=(scene_index.size, 2)) * 2.0
    return x, y, scene_index


# --------------------------------------------------------------------------- #
# Splits
# --------------------------------------------------------------------------- #

def test_no_scene_straddles_the_split(dataset):
    """Spec 7.4: the leak that inflates every number in the paper."""
    _, _, scene_index = dataset
    train, test = SP.split_frames(scene_index, probe_seed=0)
    for scene in np.unique(scene_index):
        rows = scene_index == scene
        assert train[rows].all() or test[rows].all()


def test_split_is_independent_of_enumeration_order(dataset):
    _, _, scene_index = dataset
    shuffled = np.random.default_rng(7).permutation(scene_index)
    assert np.array_equal(SP.scene_split(scene_index, 0)[0], SP.scene_split(shuffled, 0)[0])


def test_inner_folds_are_grouped_by_scene(dataset):
    """An ungrouped inner CV picks the alpha that best memorises scenes."""
    _, _, scene_index = dataset
    covered = set()
    for train_rows, val_rows in SP.grouped_folds(scene_index, 5, 0):
        assert not (set(scene_index[train_rows]) & set(scene_index[val_rows]))
        covered |= set(scene_index[val_rows])
    assert covered == set(np.unique(scene_index))


# --------------------------------------------------------------------------- #
# Ridge
# --------------------------------------------------------------------------- #

def test_ridge_matches_sklearn_exactly(dataset):
    """The eigendecomposition path must be arithmetic, not approximation."""
    sklearn_linear = pytest.importorskip("sklearn.linear_model")
    x, y, scene_index = dataset
    model = MultiRidgeCV(alphas=[10.0], n_folds=3).fit(x, y, scene_index)
    standardised = (x - x.mean(axis=0)) / x.std(axis=0)
    for k in range(y.shape[1]):
        reference = sklearn_linear.Ridge(alpha=10.0).fit(standardised, y[:, k])
        assert np.allclose(model.coef_[:, k], reference.coef_, atol=1e-8)
        assert np.allclose(model.intercept_[k], reference.intercept_, atol=1e-8)


def test_ridge_regularises_a_noise_target_hardest(dataset):
    x, _, scene_index = dataset
    rng = np.random.default_rng(1)
    signal = x @ rng.normal(size=x.shape[1])
    noise = rng.normal(size=x.shape[0])
    model = MultiRidgeCV().fit(x, np.column_stack([signal, noise]), scene_index)
    assert model.best_alpha_[1] > model.best_alpha_[0]
    assert model.best_alpha_[1] == RIDGE_ALPHAS.max()


def test_constant_feature_column_does_not_produce_nans(dataset):
    """A random-init encoder can have a dead unit."""
    x, y, scene_index = dataset
    x = x.copy()
    x[:, 0] = 3.0
    model = MultiRidgeCV(alphas=[1.0], n_folds=3).fit(x, y, scene_index)
    assert np.isfinite(model.predict(x)).all()


# --------------------------------------------------------------------------- #
# Selectivity control
# --------------------------------------------------------------------------- #

def test_control_preserves_the_marginal(dataset):
    _, _, scene_index = dataset
    labels = np.repeat(np.random.default_rng(2).normal(size=200), 10)
    shuffled = shuffle_labels_across_scenes(labels, scene_index, 0)
    assert np.allclose(np.sort(shuffled), np.sort(labels))


def test_control_breaks_a_stereotyped_within_scene_profile():
    """The bug the pilot caught (DEVIATIONS.md #5).

    Every scene shares the phase profile [0,0,0,1,1,1,1,0,0,0], so permuting
    whole blocks between scenes is a no-op and the control silently re-runs the
    real task.
    """
    scene_index = np.repeat(np.arange(150), 10)
    profile = np.array([0, 0, 0, 1, 1, 1, 1, 0, 0, 0], dtype=float)
    labels = np.tile(profile, 150)
    shuffled = shuffle_labels_across_scenes(labels, scene_index, 0)
    assert np.allclose(np.sort(shuffled), np.sort(labels))          # marginal intact
    assert (shuffled != labels).mean() > 0.2, "control left the labels where they were"


def test_control_keeps_per_scene_targets_recoverable_by_memorisation():
    """The control must still be able to detect a split leak."""
    scene_index = np.repeat(np.arange(150), 10)
    labels = np.repeat(np.random.default_rng(3).normal(size=150), 10)
    shuffled = shuffle_labels_across_scenes(labels, scene_index, 0)
    for scene in np.unique(scene_index):
        assert len(set(shuffled[scene_index == scene])) == 1


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #

def test_metrics_match_sklearn():
    metrics = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(0)
    y = rng.normal(size=400)
    p = y + rng.normal(size=400) * 0.5
    assert abs(ST.r2_score(y, p) - metrics.r2_score(y, p)) < 1e-12
    yb = rng.integers(0, 2, 400)
    pb = rng.integers(0, 2, 400)
    scores = np.round(rng.normal(size=400), 1)          # deliberate ties
    assert abs(ST.balanced_accuracy(yb, pb) - metrics.balanced_accuracy_score(yb, pb)) < 1e-12
    assert abs(ST.auroc(yb, scores) - metrics.roc_auc_score(yb, scores)) < 1e-12


@pytest.mark.parametrize("metric", ["r2", "balanced_accuracy", "auroc"])
def test_fast_bootstrap_equals_literal_resampling(metric):
    """The sufficient-statistic path is algebraically identical, not approximate."""
    rng = np.random.default_rng(0)
    n_scenes, per_scene = 60, 10
    scene_index = np.repeat(np.arange(n_scenes), per_scene)
    y = rng.normal(size=scene_index.size)
    p = y * 0.7 + rng.normal(size=scene_index.size) * 0.6

    if metric == "r2":
        truth, prediction = y, p
    elif metric == "balanced_accuracy":
        truth, prediction = (y > 0).astype(int), (p > 0).astype(int)
    else:
        truth, prediction = (y > 0).astype(int), p

    n_boot = 200
    multiplicities = ST.group_multiplicities(scene_index, n_boot, 1)
    _, fast = ST.bootstrap_metric(metric, truth, prediction, scene_index, n_boot, 1,
                                  multiplicities, return_samples=True)

    counts, _ = multiplicities
    literal = []
    for b in range(n_boot):
        rows = np.concatenate([np.flatnonzero(scene_index == s).repeat(int(counts[b, s]))
                               for s in range(n_scenes)])
        literal.append(ST.METRIC_FUNCTIONS[metric](truth[rows], prediction[rows]))
    assert np.allclose(fast, np.asarray(literal), atol=1e-10, equal_nan=True)


def test_holm_is_monotone_and_bounded():
    adjusted = ST.holm([0.01, 0.04, 0.03, 0.5])
    assert (adjusted >= np.array([0.01, 0.04, 0.03, 0.5])).all()
    assert (adjusted <= 1.0).all()


def test_tost_declares_equivalence_only_for_a_small_difference():
    rng = np.random.default_rng(0)
    small = ST.tost_bootstrap(rng.normal(0.005, 0.01, 2000), margin=0.05)
    large = ST.tost_bootstrap(rng.normal(0.20, 0.02, 2000), margin=0.05)
    assert small["equivalent"]
    assert not large["equivalent"]


def test_bootstrap_ci_brackets_the_point_estimate():
    rng = np.random.default_rng(0)
    scene_index = np.repeat(np.arange(80), 10)
    y = rng.normal(size=800)
    p = y * 0.8 + rng.normal(size=800) * 0.3
    result = ST.bootstrap_r2(y, p, scene_index, 500)
    assert result.ci_low <= result.value <= result.ci_high


def test_logistic_selects_on_balanced_accuracy_under_imbalance():
    """Accuracy would reward predicting the majority class everywhere."""
    rng = np.random.default_rng(0)
    scene_index = np.repeat(np.arange(120), 10)
    x = rng.normal(size=(1200, 16))
    y = (x @ rng.normal(size=16) + rng.normal(size=1200) * 0.5 > 1.4).astype(int)
    probe = LogisticProbe(n_folds=3).fit(x, y, scene_index)
    assert probe.cv_scores_.max() > 0.6
    assert len(set(probe.predict(x))) == 2
