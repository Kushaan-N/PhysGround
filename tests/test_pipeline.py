"""End-to-end: generation, ground truth, extraction, alignment, gates."""

from __future__ import annotations

import numpy as np
import pytest

from physground import features as FT
from physground import gates as G
from physground import paths as P
from physground.rollout import N_FRAMES, generate_scene


@pytest.fixture(scope="module")
def small_corpus(data_root, renderer):
    """A corpus small enough to build inside a test, large enough to check shape."""
    records = []
    for condition, count in (("base", 14), ("occluded", 6)):
        for index in range(count):
            records.append(generate_scene(index, 0, condition, P.scene_dir(condition, index),
                                          renderer))
    return records


def test_generation_produces_expected_labels(small_corpus):
    base = [r for r in small_corpus if r["condition"] == "base"]
    contact = np.mean([r["quality"]["n_contact_frames"] for r in base]) / N_FRAMES
    assert 0.25 <= contact <= 0.55, f"contact positive rate {contact:.3f} outside G3's band"
    assert all(r["quality"]["contact_detected"] for r in base)
    assert not any(r["quality"]["distractor_touched"] for r in base), \
        "distractor was contacted; it is only a presence control if never touched"
    assert all(r["quality"]["all_frames_in_frame"] for r in base)


def test_matched_pairs_share_their_factors(small_corpus):
    base = {r["scene_index"]: r for r in small_corpus if r["condition"] == "base"}
    occluded = {r["scene_index"]: r for r in small_corpus if r["condition"] == "occluded"}
    for index in occluded:
        for key, value in base[index]["factors"].items():
            if key in ("condition", "scene_id"):
                continue
            assert occluded[index]["factors"][key] == value


def test_occlusion_only_appears_in_the_occluded_condition(small_corpus):
    base = FT.load_targets("base")
    occluded = FT.load_targets("occluded")
    assert base["occlusion_fraction"].mean() < 0.05
    assert occluded["occlusion_fraction"].mean() > 0.15


def test_generation_is_idempotent(small_corpus, renderer):
    from physground.rollout import scene_is_complete
    assert scene_is_complete(P.scene_dir("base", 0), 0, 0, "base")


def test_generation_is_deterministic(small_corpus, renderer, tmp_path):
    """Spec 0.5: same (index, seed) must give the same scene."""
    again = generate_scene(3, 0, "base", tmp_path / "again", renderer)
    original = np.load(P.scene_dir("base", 3) / "gt.npz")
    repeat = np.load(tmp_path / "again" / "gt.npz")
    for key in ("contact_state", "support_state", "obj_pos_x", "mass", "friction_slide"):
        assert np.allclose(original[key].astype(float), repeat[key].astype(float), atol=1e-6), key
    assert again["quality"]["n_contact_frames"] > 0


def test_extraction_and_alignment(small_corpus):
    report = FT.extract("raw_pixel", "base", 0, 1, store_patches=False, progress_every=0)
    assert report["n_rows"] == 14 * N_FRAMES

    x, targets, scene_index = FT.load_dataset("raw_pixel", "base", 0, "mean")
    assert x.shape == (14 * N_FRAMES, 1024)
    assert np.array_equal(targets["scene_index"], scene_index)
    # Rows must be sorted by (scene, frame), which is what makes two separately
    # written files line up without a join.
    order = np.lexsort((targets["frame_index"], targets["scene_index"]))
    assert np.array_equal(order, np.arange(order.size))


def test_extraction_skips_completed_shards(small_corpus):
    FT.extract("raw_pixel", "base", 0, 1, store_patches=False, progress_every=0)
    again = FT.extract("raw_pixel", "base", 0, 1, store_patches=False, progress_every=0)
    assert again["skipped"] and again["reason"] == "already complete"


def test_load_dataset_refuses_misaligned_inputs(small_corpus, monkeypatch):
    """A silent misalignment would train on shuffled labels and look like a null."""
    real = FT.load_targets

    def truncated(condition, indices=None, root=None):
        table = real(condition, indices, root)
        return {k: v[:-N_FRAMES] for k, v in table.items()}

    monkeypatch.setattr(FT, "load_targets", truncated)
    with pytest.raises(RuntimeError, match="do not correspond"):
        FT.load_dataset("raw_pixel", "base", 0, "mean")


def test_gate_g3_on_the_generated_corpus(small_corpus):
    targets = FT.load_targets("base")
    report = G.gate_g3(targets, targets["scene_index"])
    failures = [c for c in report["checks"] if not c["passed"]]
    # KS tests need more than 14 scenes to be meaningful; ignore those here.
    real_failures = [c for c in failures if not c["name"].startswith("ks:")]
    assert not real_failures, real_failures


def test_gate_g2_returns_no_automated_verdict(small_corpus, data_root):
    report = G.gate_g2([("base", i) for i in range(6)], data_root / "sheet.png", n_frames=6)
    assert report["passed"] is None, "G2 must not claim a verdict a human has not given"
    assert (data_root / "sheet.png").exists()
