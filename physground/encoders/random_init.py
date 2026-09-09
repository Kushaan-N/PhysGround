"""Randomly initialised ViT-B/14, the pretraining baseline (spec 8.4.1).

Untrained ViT features are a surprisingly strong probe target: random projections
of an image preserve a great deal of linearly decodable structure, and object
position in particular survives them well. Without this control, a probe result
shows only that the *architecture plus the image* carries the property, not that
pretraining put it there -- and "DINOv2 encodes object position" would be a claim
about 224x224 images, not about DINOv2.

Same architecture, same layers, same preprocessing, same pooling as
:mod:`physground.encoders.dinov2`. The only difference is the weights, which is
what makes the comparison attributable to pretraining alone.
"""

from __future__ import annotations

from .base import register
from .dinov2 import DEFAULT_LAYERS, DinoV2Encoder

__all__ = ["RandomInitEncoder"]


class RandomInitEncoder(DinoV2Encoder):
    name = "random_b"

    def __init__(self, layers=DEFAULT_LAYERS, device: str | None = None,
                 dtype: str = "float32", arch: str = "dinov2_vitb14", seed: int = 0):
        # No patch tokens: the spatial contact probe is defined on the
        # pretrained encoder (spec 8.3), and caching another ~31 GB to answer a
        # question nobody asked would consume a third of the storage budget.
        super().__init__(layers=layers, patch_layers=(), device=device, dtype=dtype,
                         arch=arch, pretrained=False, seed=seed)


@register("random_b")
def _factory(**kwargs) -> RandomInitEncoder:
    return RandomInitEncoder(**kwargs)
