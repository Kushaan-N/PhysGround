"""Encoder interface and registry (spec 12).

Every encoder exposes one method, so the probe stage never learns which model
produced a feature matrix. That is not merely tidy: the whole argument of the
paper is a comparison across encoders, and any place where the analysis branches
on encoder identity is a place a difference in handling could masquerade as a
difference in representation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Iterable

import numpy as np

__all__ = ["Encoder", "register", "get_encoder", "available_encoders", "IMAGENET_MEAN", "IMAGENET_STD"]

#: ImageNet statistics. DINOv2, VideoMAE, and V-JEPA 2 all normalise with these;
#: each encoder module nonetheless reads them from its own checkpoint's
#: preprocessor config where one exists, rather than relying on this constant.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class Encoder(ABC):
    """Frozen feature extractor.

    Attributes
    ----------
    name
        Registry key, and the directory features are cached under.
    layers
        Blocks to extract. Spec 7.2: four evenly spaced blocks plus the final one.
    is_video
        Image encoders receive ``(N, H, W, 3)`` and return one embedding per
        frame. Video encoders receive ``(N, T, H, W, 3)`` and return one
        embedding per *scene* (spec 12), so per-frame targets are not probed on
        them in v1.
    views
        Which pooled views this encoder provides. Most give ``("cls", "mean")``;
        an encoder with no class token gives ``("mean",)`` rather than
        fabricating one, since a duplicated column would silently double a
        cell's row count in the results grid.
    """

    name: str = "encoder"
    layers: tuple[int, ...] = ()
    is_video: bool = False
    views: tuple[str, ...] = ("cls", "mean")
    #: Layers for which full patch tokens are cached. Empty for most encoders --
    #: patch tokens are ~60x the pooled payload and are needed only by the
    #: spatial contact probe (spec 7.3, 8.3).
    patch_layers: tuple[int, ...] = ()

    @abstractmethod
    def embed(self, frames: np.ndarray) -> dict[int, dict[str, np.ndarray]]:
        """Embed a batch.

        Parameters
        ----------
        frames
            ``(N, H, W, 3)`` uint8 for image encoders, ``(N, T, H, W, 3)`` uint8
            for video encoders.

        Returns
        -------
        ``{layer: {"cls": (N, D), "mean": (N, D), "patch": (N, P, D) | None}}``
        """

    def close(self) -> None:
        """Release any GPU memory. Safe to call more than once."""


_REGISTRY: dict[str, Callable[..., Encoder]] = {}


def register(name: str) -> Callable[[Callable[..., Encoder]], Callable[..., Encoder]]:
    def decorator(factory: Callable[..., Encoder]) -> Callable[..., Encoder]:
        _REGISTRY[name] = factory
        return factory
    return decorator


def get_encoder(name: str, **kwargs) -> Encoder:
    """Construct an encoder by registry key.

    Imports the concrete modules lazily so that a CPU-only corpus-generation
    container never imports torch, and a run that touches only DINOv2 never
    imports transformers (spec 7.1 stages the video encoders last).
    """
    if name not in _REGISTRY:
        from . import dinov2, random_init, raw_pixel  # noqa: F401
        if name in ("videomae_b", "vjepa2"):
            from . import videomae, vjepa2  # noqa: F401
    if name not in _REGISTRY:
        raise KeyError(f"unknown encoder {name!r}; available: {sorted(_REGISTRY)}")
    return _REGISTRY[name](**kwargs)


def available_encoders() -> list[str]:
    from . import dinov2, random_init, raw_pixel  # noqa: F401
    try:
        from . import videomae, vjepa2  # noqa: F401
    except ImportError:
        pass
    return sorted(_REGISTRY)


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #

def to_normalised_tensor(frames: np.ndarray, mean: Iterable[float], std: Iterable[float],
                         size: int | None = None, device: str = "cpu"):
    """uint8 ``(N, H, W, 3)`` to normalised float ``(N, 3, H, W)``.

    Resizing uses bilinear interpolation with ``antialias=True``. Without
    antialiasing, downsampling aliases high-frequency texture into low
    frequencies, which changes the features an encoder reports for reasons that
    have nothing to do with the scene.
    """
    import torch

    tensor = torch.as_tensor(np.ascontiguousarray(frames))
    tensor = tensor.permute(0, 3, 1, 2).float().div_(255.0)
    if size is not None and tensor.shape[-1] != size:
        tensor = torch.nn.functional.interpolate(
            tensor, size=(size, size), mode="bilinear", align_corners=False, antialias=True)
    mean_t = torch.tensor(list(mean)).view(1, 3, 1, 1)
    std_t = torch.tensor(list(std)).view(1, 3, 1, 1)
    return ((tensor - mean_t) / std_t).to(device)


def pool_tokens(patch_tokens, class_token=None) -> dict[str, np.ndarray]:
    """Build the cached pooled views from a block's output."""
    out: dict[str, np.ndarray] = {"mean": patch_tokens.mean(dim=1).float().cpu().numpy()}
    if class_token is not None:
        out["cls"] = class_token.float().cpu().numpy()
    return out
