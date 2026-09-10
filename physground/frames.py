"""PNG packing for captured frames.

Separate from :mod:`physground.render` because that module imports mujoco, and
mujoco resolves ``MUJOCO_GL`` at import time. Decoding a PNG needs neither.
Keeping them together meant feature extraction and the gate contact sheet -- both
of which only read frames back -- required a working GL backend, and failed on a
machine with no valid one for a reason unrelated to what they were doing.
"""

from __future__ import annotations

import io
from typing import Sequence

import numpy as np

__all__ = ["pack_frames", "unpack_frames", "PNG_COMPRESS_LEVEL"]

#: zlib level for frame PNGs. Measured on 224x224 renders, per 10-frame scene:
#:
#:     level 1   6.8 ms   28.2 KB/frame
#:     level 3   8.3 ms   15.7 KB/frame
#:     level 6  15.9 ms   12.1 KB/frame   <- PIL's default
#:     level 9 127.3 ms   11.5 KB/frame
#:
#: Level 3 is the knee. PIL's default would nearly double encoding time -- which
#: at 8 ms is already a third of the per-scene budget -- to save 23% of a payload
#: that totals well under a gigabyte either way.
PNG_COMPRESS_LEVEL = 3

def pack_frames(frames: Sequence[np.ndarray]) -> dict[str, np.ndarray]:
    """PNG-encode a scene's frames into arrays for a single ``.npz``.

    Spec 12 lays out ``corpus/{condition}/{scene_id}/frames/*.png``. One file
    per frame would make 40,000 files; one archive per scene makes 4,000, which
    matters because the corpus lives on a Modal volume or a network filesystem
    and the feature-extraction pass reads every frame exactly once. Fewer, larger
    reads are markedly faster there, and per-scene granularity still keeps
    parallel writers off each other's paths (spec 17.6).

    Frames are stored as concatenated PNG bytes plus an offset index rather than
    as an object array, so loading never needs ``allow_pickle`` -- which would
    otherwise make every corpus file capable of executing code on read.
    """
    from PIL import Image

    blobs = []
    for frame in frames:
        buffer = io.BytesIO()
        Image.fromarray(np.ascontiguousarray(frame)).save(
            buffer, format="PNG", compress_level=PNG_COMPRESS_LEVEL)
        blobs.append(buffer.getvalue())

    offsets = np.zeros(len(blobs) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum([len(b) for b in blobs])
    return {
        "png_data": np.frombuffer(b"".join(blobs), dtype=np.uint8),
        "png_offsets": offsets,
    }


def unpack_frames(archive) -> np.ndarray:
    """Inverse of :func:`pack_frames`. Returns ``(N, H, W, 3)`` uint8."""
    from PIL import Image

    data = np.asarray(archive["png_data"], dtype=np.uint8)
    offsets = np.asarray(archive["png_offsets"], dtype=np.int64)
    raw = data.tobytes()
    return np.stack([
        np.asarray(Image.open(io.BytesIO(raw[offsets[i]:offsets[i + 1]])).convert("RGB"))
        for i in range(len(offsets) - 1)
    ])
