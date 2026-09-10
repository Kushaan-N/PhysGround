"""Feature extraction driver: sharding, caching, fp16, manifests (spec 7.3).

Row ordering
------------
Features and targets live in different files written by different jobs, so they
need an ordering that does not depend on either job's iteration order. Every
loader here returns rows sorted by ``(scene_index, frame_index)``, and
:func:`load_dataset` asserts the two agree element-wise before handing anything
to a probe. A silent misalignment would not crash: it would train a probe on
shuffled labels and report a plausible near-chance number, which for several
targets in this project is the *expected* result and would therefore never be
questioned.

Precision
---------
Cached as fp16 (spec 3.6). Gate G4 verifies on the pilot that this costs less
than 0.01 R2 on the positive control, after which the question is settled.
Probes cast to float64 on load, so the reduced precision affects storage only,
never the linear algebra.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from . import paths as P

__all__ = [
    "scene_indices_for",
    "shard_scenes",
    "extract",
    "load_features",
    "load_targets",
    "load_dataset",
    "clear_target_cache",
    "TARGETS",
    "PER_SCENE_TARGETS",
    "REGRESSION_TARGETS",
    "CLASSIFICATION_TARGETS",
]

#: Spec 8.1. ``mass`` is probed in log space because it is sampled log-uniformly.
REGRESSION_TARGETS = ("log_mass", "friction_slide", "obj_pos_x", "obj_pos_y",
                      "obj_speed", "ee_obj_dist")
#: Spec 8.2.
CLASSIFICATION_TARGETS = ("contact_state", "support_state")
TARGETS = REGRESSION_TARGETS + CLASSIFICATION_TARGETS
#: Constant within a scene, so these are the only targets a video encoder --
#: which emits one embedding per scene (spec 12) -- can be probed on.
PER_SCENE_TARGETS = ("log_mass", "friction_slide")


# --------------------------------------------------------------------------- #
# Scene enumeration
# --------------------------------------------------------------------------- #

def scene_indices_for(condition: str, root: Path | None = None) -> list[int]:
    """Indices of every generated scene in a condition, sorted."""
    directory = (Path(root) / "corpus" / condition) if root else P.corpus_dir(condition)
    if not directory.exists():
        return []
    found = []
    for child in directory.iterdir():
        if child.is_dir() and (child / "gt.npz").exists() and (child / "frames.npz").exists():
            try:
                found.append(int(child.name))
            except ValueError:
                continue
    return sorted(found)


def shard_scenes(indices: Sequence[int], shard: int, n_shards: int) -> list[int]:
    """Contiguous-stride assignment of scenes to a shard.

    Strided rather than blocked so every shard draws scenes from across the
    corpus. Blocked assignment would give each shard a contiguous index range,
    and since generation may still be in flight, the last shard could see only
    partially written scenes while the first saw a complete set -- making a
    partial corpus look like a systematic difference between shards.
    """
    return list(np.asarray(indices)[shard::n_shards])


# --------------------------------------------------------------------------- #
# Extraction
# --------------------------------------------------------------------------- #

def _load_scene_frames(condition: str, index: int, root: Path | None = None) -> np.ndarray:
    from .frames import unpack_frames

    directory = (Path(root) / "corpus" / condition / f"{index:05d}") if root else P.scene_dir(condition, index)
    with np.load(directory / "frames.npz") as archive:
        return unpack_frames(archive)


def extract(encoder_name: str, condition: str = "base", shard: int = 0, n_shards: int = 1,
            root: Path | None = None, batch_size: int = 64, dtype: str = "float16",
            store_patches: bool = True, force: bool = False, encoder_kwargs: dict | None = None,
            progress_every: int = 20) -> dict:
    """Extract and cache features for one shard.

    Idempotent (spec 0.6): a shard whose output matches the current config hash
    is skipped, so a preempted array job re-runs only what it lost.
    """
    from .encoders import get_encoder

    data_root = Path(root) if root else None
    all_indices = scene_indices_for(condition, data_root)
    if not all_indices:
        raise FileNotFoundError(f"no generated scenes found for condition {condition!r}")
    indices = shard_scenes(all_indices, shard, n_shards)
    if not indices:
        return {"encoder": encoder_name, "shard": shard, "skipped": True, "reason": "empty shard"}

    def out_path(fn):
        path = fn(encoder_name, condition, shard)
        return (data_root / path.relative_to(P.data_root())) if data_root else path

    pooled_path = out_path(P.feature_shard)
    patch_path = out_path(P.patch_shard)

    config = {"encoder": encoder_name, "condition": condition, "shard": shard,
              "n_shards": n_shards, "dtype": dtype, "store_patches": bool(store_patches),
              "n_scenes": len(indices), "encoder_kwargs": encoder_kwargs or {}}
    cfg_hash = P.config_hash(config)
    if not force and P.is_complete(pooled_path, cfg_hash=cfg_hash):
        return {"encoder": encoder_name, "shard": shard, "skipped": True, "reason": "already complete"}

    encoder = get_encoder(encoder_name, **(encoder_kwargs or {}))
    np_dtype = np.dtype(dtype)
    started = time.perf_counter()

    pooled: dict[str, list[np.ndarray]] = {}
    patches: dict[str, list[np.ndarray]] = {}
    scene_column: list[int] = []
    frame_column: list[int] = []

    try:
        for position in range(0, len(indices), batch_size if not encoder.is_video else max(batch_size // 10, 1)):
            if encoder.is_video:
                chunk = indices[position:position + max(batch_size // 10, 1)]
                clips = np.stack([_load_scene_frames(condition, i, data_root) for i in chunk])
                outputs = encoder.embed(clips)
                scene_column.extend(chunk)
                # Video encoders emit one embedding per scene, so there is no
                # frame to point at; -1 marks that explicitly rather than
                # implying frame 0 (spec 12).
                frame_column.extend([-1] * len(chunk))
            else:
                chunk = indices[position:position + batch_size]
                frames, scenes, frame_ids = [], [], []
                for i in chunk:
                    scene_frames = _load_scene_frames(condition, i, data_root)
                    frames.append(scene_frames)
                    scenes.extend([i] * len(scene_frames))
                    frame_ids.extend(range(len(scene_frames)))
                outputs = encoder.embed(np.concatenate(frames))
                scene_column.extend(scenes)
                frame_column.extend(frame_ids)

            for layer, views in outputs.items():
                for view in encoder.views:
                    if view not in views:
                        continue
                    pooled.setdefault(f"L{layer}_{view}", []).append(
                        np.asarray(views[view], dtype=np_dtype))
                if store_patches and views.get("patch") is not None:
                    patches.setdefault(f"L{layer}_patch", []).append(
                        np.asarray(views["patch"], dtype=np_dtype))

            if progress_every and (position // max(batch_size, 1)) % progress_every == 0:
                done = min(position + batch_size, len(indices))
                print(f"[extract] {encoder_name}/{condition} shard {shard}: "
                      f"{done}/{len(indices)} scenes", flush=True)
    finally:
        encoder.close()

    arrays: dict[str, np.ndarray] = {key: np.concatenate(chunks) for key, chunks in pooled.items()}
    scene_index = np.asarray(scene_column, dtype=np.int64)
    frame_index = np.asarray(frame_column, dtype=np.int64)

    order = np.lexsort((frame_index, scene_index))
    arrays = {key: value[order] for key, value in arrays.items()}
    arrays["scene_index"] = scene_index[order]
    arrays["frame_index"] = frame_index[order]

    P.atomic_write_npz(pooled_path, arrays)
    P.write_done_marker(pooled_path, cfg_hash=cfg_hash,
                        extra={"n_rows": int(order.size), "keys": sorted(arrays)})

    if patches:
        patch_arrays = {key: np.concatenate(chunks)[order] for key, chunks in patches.items()}
        patch_arrays["scene_index"] = arrays["scene_index"]
        patch_arrays["frame_index"] = arrays["frame_index"]
        P.atomic_write_npz(patch_path, patch_arrays)
        P.write_done_marker(patch_path, cfg_hash=cfg_hash)

    elapsed = time.perf_counter() - started
    return {"encoder": encoder_name, "condition": condition, "shard": shard,
            "n_scenes": len(indices), "n_rows": int(order.size), "seconds": elapsed,
            "keys": sorted(k for k in arrays if k.startswith("L")), "skipped": False}


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #

def load_features(encoder: str, condition: str, layer: int, view: str,
                  root: Path | None = None, dtype=np.float64) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load one (layer, view) across every shard. Returns ``(x, scene_index, frame_index)``.

    NumPy's ``.npz`` is a zip archive read lazily per member, so pulling one
    layer does not decompress the others.
    """
    directory = (Path(root) / "features" / encoder / condition) if root else P.features_dir(encoder, condition)
    shards = sorted(directory.glob("shard_*.npz"))
    shards = [s for s in shards if not s.name.endswith("_patch.npz")]
    if not shards:
        raise FileNotFoundError(f"no feature shards under {directory}")

    key = f"L{layer}_{view}"
    blocks, scenes, frames = [], [], []
    for shard in shards:
        with np.load(shard) as archive:
            if key not in archive:
                raise KeyError(f"{shard.name} has no {key}; available: {sorted(archive.files)}")
            blocks.append(np.asarray(archive[key], dtype=dtype))
            scenes.append(np.asarray(archive["scene_index"]))
            frames.append(np.asarray(archive["frame_index"]))

    x = np.concatenate(blocks)
    scene_index = np.concatenate(scenes)
    frame_index = np.concatenate(frames)
    order = np.lexsort((frame_index, scene_index))
    _assert_no_duplicate_rows(scene_index[order], frame_index[order], directory)
    return x[order], scene_index[order], frame_index[order]


def _assert_no_duplicate_rows(scene_index: np.ndarray, frame_index: np.ndarray,
                              directory: Path) -> None:
    """Refuse to return a shard set that covers some rows twice.

    Shard files are named by index, so re-extracting with a *smaller*
    ``--n-shards`` leaves the surplus files from the previous run behind. The
    loader globs the directory, so those stale shards are concatenated with the
    fresh ones and every row they cover appears twice -- silently. Duplicated
    rows land on both sides of a scene-level split (they belong to the same
    scene, so the split itself stays honest) but they reweight the training set
    and inflate the apparent sample size in every bootstrap.
    """
    if scene_index.size == 0:
        return
    pairs = np.stack([scene_index, frame_index], axis=1)
    duplicated = np.zeros(scene_index.size, dtype=bool)
    duplicated[1:] = (pairs[1:] == pairs[:-1]).all(axis=1)
    if duplicated.any():
        example = pairs[np.flatnonzero(duplicated)[0]]
        raise RuntimeError(
            f"{directory} yields duplicate rows (e.g. scene {example[0]} frame {example[1]} "
            f"appears more than once across {scene_index.size} rows). This usually means stale "
            "shard files from a run with a different --n-shards. Delete the directory and "
            "re-extract.")


def load_patches(encoder: str, condition: str, layer: int, root: Path | None = None,
                 dtype=np.float32) -> tuple[np.ndarray, np.ndarray]:
    """Load cached patch tokens for one layer. Returns ``(x, scene_index)``."""
    directory = (Path(root) / "features" / encoder / condition) if root else P.features_dir(encoder, condition)
    shards = sorted(directory.glob("shard_*_patch.npz"))
    if not shards:
        raise FileNotFoundError(f"no patch shards under {directory}")
    key = f"L{layer}_patch"
    blocks, scenes, frames = [], [], []
    for shard in shards:
        with np.load(shard) as archive:
            blocks.append(np.asarray(archive[key], dtype=dtype))
            scenes.append(np.asarray(archive["scene_index"]))
            frames.append(np.asarray(archive["frame_index"]))
    x = np.concatenate(blocks)
    scene_index = np.concatenate(scenes)
    frame_index = np.concatenate(frames)
    order = np.lexsort((frame_index, scene_index))
    _assert_no_duplicate_rows(scene_index[order], frame_index[order], directory)
    return x[order], scene_index[order]


#: Memoised ground truth, keyed by (condition, root, scene indices).
#:
#: Reading it costs 1.42 s for a 1500-scene corpus, because it opens one
#: ``gt.npz`` per scene, and the probe grid calls it twice per cell across
#: dozens of cells -- tens of thousands of file opens to re-read something that
#: cannot have changed. The cache assumes the corpus is immutable for the life
#: of the process, which holds because generation and probing are separate
#: stages; the scene list is part of the key, so a corpus that grew between
#: calls misses rather than returning a stale answer.
_TARGET_CACHE: dict[tuple, dict[str, np.ndarray]] = {}


def clear_target_cache() -> None:
    """Drop the memoised ground truth. For tests that regenerate a corpus."""
    _TARGET_CACHE.clear()


def load_targets(condition: str, indices: Iterable[int] | None = None,
                 root: Path | None = None) -> dict[str, np.ndarray]:
    """Load per-frame ground truth, sorted by ``(scene_index, frame_index)``.

    Returns a fresh dict each call so a caller cannot corrupt the cache by
    reassigning a key; the arrays inside are shared and must be treated as
    read-only.
    """
    data_root = Path(root) if root else None
    wanted = list(indices) if indices is not None else scene_indices_for(condition, data_root)

    cache_key = (condition, str(data_root), tuple(wanted))
    cached = _TARGET_CACHE.get(cache_key)
    if cached is not None:
        return dict(cached)

    columns: dict[str, list[np.ndarray]] = {}
    scene_column: list[np.ndarray] = []
    for index in wanted:
        directory = (data_root / "corpus" / condition / f"{index:05d}") if data_root \
            else P.scene_dir(condition, index)
        with np.load(directory / "gt.npz") as archive:
            n = len(archive["frame_index"])
            for key in archive.files:
                if key == "phase":
                    continue
                columns.setdefault(key, []).append(np.asarray(archive[key]))
        scene_column.append(np.full(n, index, dtype=np.int64))

    out = {key: np.concatenate(values) for key, values in columns.items()}
    out["scene_index"] = np.concatenate(scene_column)
    order = np.lexsort((out["frame_index"], out["scene_index"]))
    sorted_out = {key: value[order] for key, value in out.items()}
    _TARGET_CACHE[cache_key] = sorted_out
    return dict(sorted_out)


def load_dataset(encoder: str, condition: str, layer: int, view: str,
                 root: Path | None = None) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray]:
    """Features and aligned targets. Returns ``(x, targets, scene_index)``.

    For video encoders there is one row per scene, so per-frame targets are
    collapsed by taking each scene's first frame -- valid only for targets that
    are constant within a scene, which is why :data:`PER_SCENE_TARGETS` exists
    and why the caller must respect it.
    """
    x, scene_index, frame_index = load_features(encoder, condition, layer, view, root)
    targets = load_targets(condition, root=root)

    if np.all(frame_index < 0):                      # video encoder: one row per scene
        first = np.flatnonzero(np.r_[True, targets["scene_index"][1:] != targets["scene_index"][:-1]])
        targets = {key: value[first] for key, value in targets.items()}

    if targets["scene_index"].shape != scene_index.shape or \
            not np.array_equal(targets["scene_index"], scene_index):
        raise RuntimeError(
            "feature rows and target rows do not correspond. Features cover "
            f"{scene_index.size} rows over {np.unique(scene_index).size} scenes; targets cover "
            f"{targets['scene_index'].size} rows over {np.unique(targets['scene_index']).size}. "
            "This usually means features were extracted before the corpus finished generating.")
    return x, targets, scene_index
