"""Scene-level train/test splitting and matched-pair handling (spec 7.4).

The single most consequential line of methodology in this project:

> **Split by scene ID, never by frame.**

Ten frames from one scene share lighting, object identity, colour, mass, and
friction. A frame-level split puts near-duplicates on both sides and inflates
every reported number -- most severely for exactly the per-scene properties
(``mass``, ``friction_slide``) whose *absence* is the paper's claim. A leaked
mass probe reads as evidence against H2, which is the hypothesis the paper
exists to defend.

Splitting is therefore expressed only in terms of scene indices, and frame masks
are derived from those. No function here accepts a frame-level partition.
"""

from __future__ import annotations

from typing import Iterator, Sequence

import numpy as np

__all__ = [
    "scene_split",
    "frame_mask",
    "grouped_folds",
    "split_frames",
]


def scene_split(scene_indices: Sequence[int], probe_seed: int,
                train_frac: float = 0.8) -> tuple[np.ndarray, np.ndarray]:
    """Partition scene *indices* into train and test.

    Keyed on the scene index, not the ``"{condition}_{index}"`` id, so
    ``base_00042`` and ``occluded_00042`` always land on the same side. Those two
    share every sampled factor and the entire scripted trajectory (spec 5.4), so
    separating them would let a probe train on a scene and be tested on its own
    twin -- the frame-level leak the module docstring warns about, wearing a
    different disguise.

    Sorted before shuffling so the result depends on ``probe_seed`` alone and not
    on the order the caller happened to enumerate scenes in (spec 0.5).
    """
    unique = np.array(sorted(set(int(i) for i in scene_indices)), dtype=np.int64)
    rng = np.random.default_rng(probe_seed)
    shuffled = unique.copy()
    rng.shuffle(shuffled)
    n_train = int(round(train_frac * shuffled.size))
    n_train = min(max(n_train, 1), shuffled.size - 1) if shuffled.size > 1 else shuffled.size
    return np.sort(shuffled[:n_train]), np.sort(shuffled[n_train:])


def frame_mask(frame_scene_index: np.ndarray, scenes: np.ndarray) -> np.ndarray:
    """Boolean mask over frames selecting those belonging to ``scenes``."""
    return np.isin(np.asarray(frame_scene_index), np.asarray(scenes))


def split_frames(frame_scene_index: np.ndarray, probe_seed: int,
                 train_frac: float = 0.8) -> tuple[np.ndarray, np.ndarray]:
    """Convenience wrapper: frame-level boolean masks from a scene-level split."""
    train_scenes, test_scenes = scene_split(frame_scene_index, probe_seed, train_frac)
    return frame_mask(frame_scene_index, train_scenes), frame_mask(frame_scene_index, test_scenes)


def grouped_folds(frame_scene_index: np.ndarray, n_folds: int = 5,
                  seed: int = 0) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Grouped K-fold over scenes, for the internal hyperparameter search.

    Yields ``(train_rows, val_rows)`` index arrays. Grouping matters as much here
    as in the outer split: an ungrouped inner CV picks the regularisation
    strength that best memorises scenes, which then generalises worst on the held
    out scenes -- a silent, systematic bias toward too little regularisation on
    exactly the hardest targets.

    Scenes are dealt to folds round-robin after shuffling, which keeps fold sizes
    within one scene of each other even when the scene count is not divisible by
    ``n_folds``.
    """
    frame_scene_index = np.asarray(frame_scene_index)
    unique = np.array(sorted(set(int(i) for i in frame_scene_index)), dtype=np.int64)
    if unique.size < n_folds:
        raise ValueError(f"{unique.size} scenes cannot be split into {n_folds} folds")

    rng = np.random.default_rng(seed)
    shuffled = unique.copy()
    rng.shuffle(shuffled)
    assignment = {int(scene): i % n_folds for i, scene in enumerate(shuffled)}
    fold_of_row = np.array([assignment[int(s)] for s in frame_scene_index])

    all_rows = np.arange(frame_scene_index.size)
    for fold in range(n_folds):
        validation = fold_of_row == fold
        yield all_rows[~validation], all_rows[validation]
