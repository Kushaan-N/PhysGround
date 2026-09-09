"""Raw-pixel baseline (spec 8.4.2).

Downsample to 32x32 grayscale, flatten to 1024 dims, probe identically. If the
pixel baseline matches an encoder, the encoder contributed nothing -- which is
why this baseline is mandatory rather than a nicety.

Implemented with NumPy only: it is the one "encoder" that must be available in
containers with no torch installed, and area-averaging 224 -> 32 is exact
integer block reduction, not something worth a framework.
"""

from __future__ import annotations

import numpy as np

from .base import Encoder, register

__all__ = ["RawPixelEncoder"]

#: Rec. 601 luma weights, the standard RGB-to-grayscale conversion.
_LUMA = np.array([0.299, 0.587, 0.114], dtype=np.float32)


class RawPixelEncoder(Encoder):
    name = "raw_pixel"
    layers = (0,)
    is_video = False
    views = ("mean",)

    def __init__(self, side: int = 32):
        self.side = int(side)

    def embed(self, frames: np.ndarray) -> dict[int, dict[str, np.ndarray]]:
        frames = np.asarray(frames)
        if frames.ndim == 5:  # a clip: collapse time by averaging, so the
            frames = frames.mean(axis=1)  # baseline stays comparable per scene
        gray = (frames.astype(np.float32) @ _LUMA) / 255.0        # (N, H, W)

        n, height, width = gray.shape
        if height % self.side or width % self.side:
            raise ValueError(
                f"frame size {height}x{width} is not divisible by {self.side}; "
                "block-average downsampling needs an integer ratio")
        block_h, block_w = height // self.side, width // self.side
        # Area-average by reshaping into blocks. Equivalent to an antialiased
        # box filter, and exact -- no interpolation kernel to disagree about.
        pooled = gray.reshape(n, self.side, block_h, self.side, block_w).mean(axis=(2, 4))
        return {0: {"mean": pooled.reshape(n, -1).astype(np.float32), "patch": None}}


@register("raw_pixel")
def _factory(**kwargs) -> RawPixelEncoder:
    return RawPixelEncoder(**kwargs)
