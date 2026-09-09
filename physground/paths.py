"""Deterministic output layout, atomic writes, and completion markers.

Everything this project produces lives under a single root (``outputs/`` by
default, or ``$PHYSGROUND_DATA`` — which is how the Modal path points at the
mounted volume without changing any call site).

Idempotence (spec 0.6)
----------------------
This pipeline runs as preemptible SLURM array jobs and Modal containers, so a
process can die at any instant, including mid-write. Two mechanisms handle that:

*Atomic writes.* Payloads go to ``<path>.tmp.<pid>`` and are then moved into
place with ``os.replace``, which is atomic within a filesystem. A reader never
observes a half-written file, so a truncated NumPy archive can never be mistaken
for a complete one.

*Completion markers.* Existence alone does not prove a file is the one the
current config asked for — a rerun with different parameters produces a
same-named file with different contents. Each payload therefore gets a sidecar
``.done`` JSON recording its size, its SHA-256, and the config hash that
produced it. :func:`is_complete` checks size and config hash by default (cheap,
O(1) per file, which matters when scanning 4,000 scene directories at job
start) and verifies the full digest only when asked.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np

__all__ = [
    "data_root",
    "corpus_dir",
    "scene_dir",
    "features_dir",
    "feature_shard",
    "patch_shard",
    "gates_dir",
    "results_dir",
    "figures_dir",
    "config_hash",
    "atomic_write_bytes",
    "atomic_write_json",
    "atomic_write_npz",
    "write_done_marker",
    "is_complete",
    "read_json",
]

_REPO_ROOT = Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------- #
# Layout
# --------------------------------------------------------------------------- #

def data_root() -> Path:
    """Root for all generated artifacts.

    ``$PHYSGROUND_DATA`` wins when set. That is the only thing the Modal
    functions need to override in order to write to ``/data`` (spec 17.5).
    """
    env = os.environ.get("PHYSGROUND_DATA")
    return Path(env).expanduser().resolve() if env else _REPO_ROOT / "outputs"


def corpus_dir(condition: str) -> Path:
    return data_root() / "corpus" / condition


def scene_dir(condition: str, index: int) -> Path:
    """Directory for one scene. Scene id is ``f"{condition}_{index:05d}"`` (spec 5.4)."""
    return corpus_dir(condition) / f"{index:05d}"


def features_dir(encoder: str, condition: str) -> Path:
    return data_root() / "features" / encoder / condition


def feature_shard(encoder: str, condition: str, shard: int) -> Path:
    """Pooled features (cls + mean, every layer) for one shard, in one archive.

    Spec 12 sketches ``features/{encoder}/{layer}/{shard}.npz``. We put every
    layer in one archive per shard instead: the pooled payload for a shard is a
    few tens of MB, and one file per (encoder, layer, shard) multiplies inode
    count and per-file open overhead by the layer count for no benefit, on
    exactly the network filesystems this runs on. Layers stay addressable as
    array keys (``L8_cls``), and NumPy's zip-archive format loads them lazily,
    so reading one layer still does not read the others.
    """
    return features_dir(encoder, condition) / f"shard_{shard:04d}.npz"


def patch_shard(encoder: str, condition: str, shard: int) -> Path:
    """Full patch tokens, kept in a separate archive from the pooled views.

    Patch tokens are ~60x the size of the pooled views (spec 7.3). Splitting
    them out means the probe stage, which reads pooled features for 112 of its
    113 configurations, never touches the 31 GB payload it does not need.
    """
    return features_dir(encoder, condition) / f"shard_{shard:04d}_patch.npz"


def gates_dir() -> Path:
    return data_root() / "gates"


def results_dir(exp: str, seed: int | None = None) -> Path:
    base = data_root() / "results" / exp
    return base if seed is None else base / f"seed_{seed:02d}"


def figures_dir() -> Path:
    return data_root() / "figures"


# --------------------------------------------------------------------------- #
# Config hashing
# --------------------------------------------------------------------------- #

def config_hash(config: Any, length: int = 12) -> str:
    """Stable short hash of a config.

    Uses ``sort_keys`` so dict ordering cannot change the hash, and rejects
    values JSON cannot represent rather than silently hashing a ``repr``. NumPy
    scalars are coerced, because a config assembled from sampled factors is full
    of ``np.float64`` and letting those through as ``repr`` strings would make
    the hash depend on NumPy's formatting rather than on the value.
    """
    blob = json.dumps(config, sort_keys=True, separators=(",", ":"), default=_json_default)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:length]


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"config contains a non-serialisable value of type {type(obj).__name__}")


# --------------------------------------------------------------------------- #
# Atomic writes
# --------------------------------------------------------------------------- #

def _tmp_path(path: Path) -> Path:
    # The pid suffix keeps two workers that race on the same output from
    # clobbering each other's temp file. os.replace then makes whichever
    # finishes last the winner, and both wrote identical bytes anyway.
    return path.with_name(f"{path.name}.tmp.{os.getpid()}")


def atomic_write_bytes(path: Path, payload: bytes) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp_path(path)
    try:
        with open(tmp, "wb") as fh:
            fh.write(payload)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
    return path


def atomic_write_json(path: Path, obj: Any, *, indent: int = 2) -> Path:
    blob = json.dumps(obj, indent=indent, sort_keys=True, default=_json_default)
    return atomic_write_bytes(Path(path), blob.encode("utf-8"))


def atomic_write_npz(path: Path, arrays: dict[str, np.ndarray], *, compress: bool = False) -> Path:
    """Write a ``.npz`` atomically.

    ``compress=False`` by default. The dominant payloads here are fp16 encoder
    features, which are close to incompressible; zlib would burn CPU on every
    read and write to save a few percent. Compression is worth it only for the
    small integer-and-boolean ground-truth archives, where callers opt in.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp_path(path)
    try:
        with open(tmp, "wb") as fh:
            if compress:
                np.savez_compressed(fh, **arrays)
            else:
                np.savez(fh, **arrays)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
    return path


# --------------------------------------------------------------------------- #
# Completion markers
# --------------------------------------------------------------------------- #

def _marker_path(path: Path) -> Path:
    return Path(path).with_name(Path(path).name + ".done")


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def write_done_marker(path: Path, *, cfg_hash: str, extra: dict | None = None) -> Path:
    """Record that ``path`` is complete and which config produced it."""
    path = Path(path)
    marker = {
        "size": path.stat().st_size,
        "sha256": _sha256(path),
        "config_hash": cfg_hash,
    }
    if extra:
        marker.update(extra)
    return atomic_write_json(_marker_path(path), marker)


def is_complete(path: Path, *, cfg_hash: str | None = None, verify_digest: bool = False) -> bool:
    """Whether ``path`` is a finished artifact of the given config.

    Cheap by default: existence, byte size, and config hash. The full SHA-256 is
    checked only under ``verify_digest``, because a job that starts by hashing
    every one of 4,000 existing scene archives spends its first minutes on I/O
    to re-confirm something ``os.replace`` already guaranteed.
    """
    path = Path(path)
    marker = _marker_path(path)
    if not path.exists() or not marker.exists():
        return False
    try:
        info = json.loads(marker.read_text())
    except (json.JSONDecodeError, OSError):
        return False
    if info.get("size") != path.stat().st_size:
        return False
    if cfg_hash is not None and info.get("config_hash") != cfg_hash:
        return False
    if verify_digest and info.get("sha256") != _sha256(path):
        return False
    return True


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text())


def iter_scene_dirs(condition: str, indices: Iterable[int]) -> Iterable[Path]:
    for i in indices:
        yield scene_dir(condition, i)
