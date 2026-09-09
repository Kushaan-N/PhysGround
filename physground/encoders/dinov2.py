"""DINOv2 ViT-B/14, the project's primary encoder (spec 7.1).

12 blocks, patch 14, so a 224x224 input gives 16x16 = 256 patch tokens plus a
class token, at 768 dimensions.

The model is loaded frozen and in ``eval()`` mode, and every forward runs under
``torch.no_grad()``. The premise of the paper is that the encoder is fixed
before probing begins (spec 4), so nothing here may accumulate gradients or
update a running statistic.
"""

from __future__ import annotations

import numpy as np

from .base import IMAGENET_MEAN, IMAGENET_STD, Encoder, pool_tokens, register, to_normalised_tensor

__all__ = ["DinoV2Encoder"]

#: Spec 7.2: four evenly spaced blocks plus the final one, for 12 total blocks.
DEFAULT_LAYERS = (2, 5, 8, 11)
#: Spec 7.3: full patch tokens only for these, and only for this encoder.
#: All four layers would be ~62 GB; two is ~31 GB.
DEFAULT_PATCH_LAYERS = (8, 11)

INPUT_SIZE = 224


class DinoV2Encoder(Encoder):
    name = "dinov2_b"
    is_video = False
    views = ("cls", "mean")

    def __init__(self, layers=DEFAULT_LAYERS, patch_layers=DEFAULT_PATCH_LAYERS,
                 device: str | None = None, dtype: str = "float32",
                 arch: str = "dinov2_vitb14", pretrained: bool = True, seed: int = 0):
        import torch

        self.layers = tuple(int(v) for v in layers)
        self.patch_layers = tuple(int(v) for v in patch_layers)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._torch = torch

        if not pretrained:
            # Seed *before* construction: with pretrained=False every weight
            # comes from the RNG, so the random-init baseline is reproducible
            # only if the generator state is fixed at this point (spec 0.5).
            torch.manual_seed(seed)

        self.model = torch.hub.load("facebookresearch/dinov2", arch, pretrained=pretrained)
        self.model.eval().to(self.device)
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

        self._dtype = getattr(torch, dtype)
        if self._dtype is not torch.float32:
            self.model.to(self._dtype)

    def embed(self, frames: np.ndarray) -> dict[int, dict[str, np.ndarray]]:
        torch = self._torch
        frames = np.asarray(frames)
        if frames.ndim != 4:
            raise ValueError(f"{self.name} is an image encoder; expected (N, H, W, 3), got {frames.shape}")

        batch = to_normalised_tensor(frames, IMAGENET_MEAN, IMAGENET_STD,
                                     size=INPUT_SIZE, device=self.device).to(self._dtype)

        with torch.no_grad():
            outputs = self.model.get_intermediate_layers(
                batch, n=list(self.layers), reshape=False, return_class_token=True, norm=True)

        result: dict[int, dict[str, np.ndarray]] = {}
        for layer, (patch_tokens, class_token) in zip(self.layers, outputs):
            views = pool_tokens(patch_tokens, class_token)
            views["patch"] = (patch_tokens.float().cpu().numpy()
                              if layer in self.patch_layers else None)
            result[layer] = views
        return result

    def close(self) -> None:
        self.model = None
        if self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()


@register("dinov2_b")
def _factory(**kwargs) -> DinoV2Encoder:
    return DinoV2Encoder(**kwargs)
