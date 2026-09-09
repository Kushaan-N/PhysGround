"""Factor sampling and the decorrelation gate (spec 5.3, 9.1)."""

from __future__ import annotations

import numpy as np
import pytest

from physground import factors as F


def test_sampling_is_pure_in_index_and_seed():
    """Spec 0.5: reproducible from (scene_id, seed) alone."""
    a = F.sample_factors(17, 0)
    assert F.sample_factors(17, 0) == a
    assert F.sample_factors(17, 1) != a
    assert F.sample_factors(18, 0) != a


def test_condition_never_changes_a_sampled_value():
    """Matched occlusion pairs must share every factor (spec 5.4)."""
    base = F.sample_factors(42, 0, "base")
    occluded = F.sample_factors(42, 0, "occluded")
    for key in base:
        if key in ("condition", "scene_id"):
            continue
        assert base[key] == occluded[key], f"{key} differs between matched pair members"


def test_seeding_does_not_use_salted_hash():
    """`hash` on str is salted per process; scene sampling must not be."""
    key = F._name_key("mass")
    assert key == F._name_key("mass")
    # BLAKE2b of a fixed string is a fixed number across interpreters.
    assert isinstance(key, int) and key > 0


def test_adding_a_factor_does_not_perturb_others():
    """Per-factor streams: values must not depend on the factor table's shape."""
    before = F.sample_factors(3, 0)["mass"]
    original = F.FACTORS
    try:
        F.FACTORS = original + (F.FactorSpec("zzz_probe", "appearance", "continuous",
                                            lambda rng: float(rng.uniform())),)
        after = F.sample_factors(3, 0)["mass"]
    finally:
        F.FACTORS = original
    assert before == after


@pytest.mark.parametrize("values", [
    np.array([3.0, 1.0, 2.0]),
    np.zeros(7),
    np.array([1.0, 1.0, 2.0, 2.0, 2.0]),
])
def test_rankdata_matches_scipy(values):
    scipy_stats = pytest.importorskip("scipy.stats")
    assert np.allclose(F._rankdata(values), scipy_stats.rankdata(values))


def test_spearman_matches_scipy():
    scipy_stats = pytest.importorskip("scipy.stats")
    rng = np.random.default_rng(0)
    matrix = rng.normal(size=(200, 5))
    matrix[:, 1] = np.round(matrix[:, 1])          # heavy ties
    assert np.allclose(F.spearman_matrix(matrix), scipy_stats.spearmanr(matrix).statistic,
                       atol=1e-10)


def test_g1_passes_at_corpus_size():
    """The property the whole paper rests on."""
    report = F.check_decorrelation(F.factors_table(range(3000), 0))
    assert report["passed"], (report["max_abs_rho"], report["worst_pair"])
    assert report["max_abs_rho"] <= 0.10
    assert report["n_significant_after_holm"] == 0


def test_g1_at_pilot_size_fails_on_noise_alone():
    """Documents why G1 does not run on 50 scenes (DEVIATIONS.md #1).

    With SE ~ 1/sqrt(n-1) = 0.143 at n=50, a correctly independent pair clears
    the 0.10 threshold about half the time. If this ever starts passing, the
    deviation should be revisited -- but the significance criterion must stay
    clean either way, because nothing is actually correlated.
    """
    report = F.check_decorrelation(F.factors_table(range(50), 0))
    assert report["max_abs_rho"] > 0.10
    assert report["n_significant_after_holm"] == 0
