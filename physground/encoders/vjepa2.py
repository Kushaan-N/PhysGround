"""V-JEPA 2 ViT-L, a video encoder for H3 (spec 7.1).

Spec 7.1 warns not to write this from memory. Everything below was taken from
the live checkpoint rather than recalled, and the source of each fact is
recorded so a future reader can re-check it against the same files:

``facebook/vjepa2-vitl-fpc64-256/config.json``
    ``hidden_size`` 1024, ``num_hidden_layers`` 24, ``patch_size`` 16,
    ``tubelet_size`` 2, ``crop_size`` 256, ``frames_per_clip`` 64,
    ``architectures: ["VJEPA2Model"]``.

``facebook/vjepa2-vitl-fpc64-256/video_preprocessor_config.json``
    ``video_processor_type`` ``VJEPA2VideoProcessor``, resize to
    ``{"shortest_edge": 292}`` with ``default_to_square: true``, then
    ``do_center_crop`` to 256x256, rescale by 1/255, normalise with ImageNet
    mean/std.

``README.md``
    Load with ``AutoModel`` / ``AutoVideoProcessor``; the processor returns
    ``pixel_values_videos``; features come from ``get_vision_features``, and an
    image is handled by repeating it along time.

Two consequences worth stating plainly:

*Layer indices.* 24 blocks, so the evenly-spaced-plus-final set of spec 7.2 is
(5, 11, 17, 23), not DINOv2's (2, 5, 8, 11).

*No class token.* Like other JEPA encoders this produces a pure patch sequence,
so only the ``mean`` view is advertised (see :mod:`physground.encoders.videomae`
for why a duplicated view would be worse than a missing one).

*Clip length.* The checkpoint is trained at 64 frames; the corpus has 10 per
scene. Frames are resampled to :data:`DEFAULT_CLIP_FRAMES`, which is a genuine
departure from the training regime and is noted as a limitation rather than
hidden -- H3 asks whether a video encoder recovers dynamics *at all*, and a
short clip can only understate that.
"""

from __future__ import annotations

import numpy as np

from .base import Encoder, pick_device, register

__all__ = ["VJEPA2Encoder"]

MODEL_ID = "facebook/vjepa2-vitl-fpc64-256"
#: 24 blocks: four evenly spaced plus the final one (spec 7.2).
DEFAULT_LAYERS = (5, 11, 17, 23)
#: Must be a multiple of the checkpoint's tubelet size (2).
DEFAULT_CLIP_FRAMES = 16


class VJEPA2Encoder(Encoder):
    name = "vjepa2"
    is_video = True
    views = ("mean",)

    def __init__(self, layers=DEFAULT_LAYERS, device: str | None = None,
                 model_id: str = MODEL_ID, n_frames: int = DEFAULT_CLIP_FRAMES,
                 dtype: str = "float32"):
        import torch
        from transformers import AutoModel, AutoVideoProcessor

        self.layers = tuple(int(v) for v in layers)
        self._torch = torch
        self.device = pick_device(device)

        self.processor = AutoVideoProcessor.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id)
        self.model.eval().to(self.device)
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

        self._dtype = getattr(torch, dtype)
        if self._dtype is not torch.float32:
            self.model.to(self._dtype)

        config = self.model.config
        self.tubelet = int(getattr(config, "tubelet_size", 2))
        self.n_frames = int(n_frames)
        if self.n_frames % self.tubelet:
            raise ValueError(f"n_frames={self.n_frames} must be a multiple of tubelet {self.tubelet}")

        n_blocks = int(getattr(config, "num_hidden_layers", 24))
        if max(self.layers) >= n_blocks:
            raise ValueError(
                f"layer {max(self.layers)} requested but the checkpoint has {n_blocks} blocks. "
                "The layer set is checkpoint-specific; see spec 7.2.")

    def _resample(self, clip_length: int) -> np.ndarray:
        return np.clip(np.round(np.linspace(0, clip_length - 1, self.n_frames)), 0,
                       clip_length - 1).astype(int)

    def embed(self, frames: np.ndarray) -> dict[int, dict[str, np.ndarray]]:
        torch = self._torch
        frames = np.asarray(frames)
        if frames.ndim != 5:
            raise ValueError(f"{self.name} is a video encoder; expected (N, T, H, W, 3), got {frames.shape}")

        clip = frames[:, self._resample(frames.shape[1])]

        # Preprocess through the checkpoint's own processor rather than
        # reimplementing resize/crop/normalise. The processor is fed one clip at
        # a time because it expects a single video's frames as (T, H, W, C).
        videos = [self.processor(list(sample), return_tensors="pt")["pixel_values_videos"][0]
                  for sample in clip]
        batch = torch.stack(videos).to(self.device).to(self._dtype)

        with torch.no_grad():
            outputs = self.model(pixel_values_videos=batch, skip_predictor=True,
                                 output_hidden_states=True)

        hidden = self._hidden_states(outputs)
        result: dict[int, dict[str, np.ndarray]] = {}
        for layer in self.layers:
            tokens = hidden[layer + 1]
            result[layer] = {"mean": tokens.mean(dim=1).float().cpu().numpy(), "patch": None}
        return result

    @staticmethod
    def _hidden_states(outputs):
        """Locate per-block hidden states across output container shapes.

        V-JEPA 2 returns a composite output holding both encoder and predictor
        results, and the attribute layout has moved between transformers
        releases. Failing loudly here beats silently probing the wrong tensor:
        an encoder that returned predictor states would still produce
        plausible-looking numbers.
        """
        for attribute in ("hidden_states", "encoder_hidden_states"):
            states = getattr(outputs, attribute, None)
            if states is not None:
                return states
        encoder = getattr(outputs, "last_hidden_state", None)
        encoder_output = getattr(encoder, "hidden_states", None)
        if encoder_output is not None:
            return encoder_output
        raise RuntimeError(
            "V-JEPA 2 output exposed no hidden_states. The checkpoint's output "
            f"container is {type(outputs).__name__} with fields "
            f"{list(getattr(outputs, 'keys', lambda: [])())}. Re-read the model "
            "card before guessing (spec 7.1).")

    def close(self) -> None:
        self.model = None
        if self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()


@register("vjepa2")
def _factory(**kwargs) -> VJEPA2Encoder:
    return VJEPA2Encoder(**kwargs)
