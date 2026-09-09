"""VideoMAE base, a video encoder for H3 (spec 7.1).

12 blocks, hidden size 768, patch 16, tubelet 2. At the checkpoint's 16-frame
224x224 input that gives 8 x 14 x 14 = 1568 tokens.

VideoMAE has **no class token** -- its output is a pure patch sequence -- so this
encoder advertises only the ``mean`` view rather than duplicating the mean under
a ``cls`` key. A fabricated second view would double this encoder's row count in
the results grid with a perfectly correlated copy, which would look like extra
evidence while carrying none.

Preprocessing constants are read from the checkpoint's own processor config
rather than hard-coded, so a checkpoint revision that changed them cannot
silently invalidate the features.
"""

from __future__ import annotations

import numpy as np

from .base import Encoder, pick_device, register

__all__ = ["VideoMAEEncoder", "repair_attention_biases"]

MODEL_ID = "MCG-NJU/videomae-base"
#: Spec 7.2: four evenly spaced blocks plus the final one, for 12 blocks.
DEFAULT_LAYERS = (2, 5, 8, 11)


def _load_checkpoint_tensors(model_id: str) -> dict:
    """Fetch the checkpoint's raw tensors, preferring safetensors."""
    from huggingface_hub import hf_hub_download

    try:
        from safetensors.torch import load_file
        return load_file(hf_hub_download(model_id, "model.safetensors"))
    except Exception:
        import torch
        return torch.load(hf_hub_download(model_id, "pytorch_model.bin"),
                          map_location="cpu", weights_only=True)


def repair_attention_biases(model, model_id: str) -> dict:
    """Restore VideoMAE's attention biases, which transformers 5.x drops.

    VideoMAE follows BEiT in giving attention a learned ``q_bias`` and
    ``v_bias`` with the key bias fixed at zero, and the published checkpoint
    stores exactly those two tensors. transformers 5.x expects the standard
    ``query.bias`` / ``key.bias`` / ``value.bias`` triple, finds none of them,
    and reports::

        encoder.layer.{0...11}.attention.attention.query.bias | MISSING

    The pretrained biases are then discarded and replaced with zeros. This is
    not cosmetic: measured on ``MCG-NJU/videomae-base``, the checkpoint's
    layer-0 ``q_bias`` has norm 17.5 and ``v_bias`` norm 1.8, against the zeros
    actually loaded. Left alone, every VideoMAE feature in this project would
    come from a partly lobotomised encoder, and H3's video-versus-frame
    comparison would understate the video encoder for a reason that has nothing
    to do with video. Measured against the repaired encoder, the pooled features
    differ by 19-25% in relative L2 norm at every probed layer -- large enough to
    move a result, small enough to look like a real finding.

    The repair is a no-op on any version that loads the biases correctly: it
    runs only when the biases are all zero *and* the checkpoint supplies them.
    Raises if it cannot verify success, since a silent failure here is exactly
    the outcome being guarded against.
    """
    import torch

    layers = getattr(getattr(model, "encoder", None), "layer", None)
    if layers is None:
        return {"applied": False, "reason": "unexpected model layout"}

    def attention(block):
        return block.attention.attention

    already = any(
        getattr(attention(block), name).bias is not None
        and float(getattr(attention(block), name).bias.abs().sum()) > 0
        for block in layers for name in ("query", "value")
    )
    if already:
        return {"applied": False, "reason": "biases already loaded"}

    tensors = _load_checkpoint_tensors(model_id)
    prefixes = ("videomae.encoder.layer", "encoder.layer")
    repaired = 0
    with torch.no_grad():
        for index, block in enumerate(layers):
            block_attention = attention(block)
            for prefix in prefixes:
                q = tensors.get(f"{prefix}.{index}.attention.attention.q_bias")
                v = tensors.get(f"{prefix}.{index}.attention.attention.v_bias")
                if q is None or v is None:
                    continue
                block_attention.query.bias.copy_(q.to(block_attention.query.bias.dtype))
                block_attention.value.bias.copy_(v.to(block_attention.value.bias.dtype))
                if block_attention.key.bias is not None:
                    block_attention.key.bias.zero_()   # VideoMAE has no key bias
                repaired += 1
                break

    if repaired != len(layers):
        raise RuntimeError(
            f"VideoMAE attention biases are zero and only {repaired}/{len(layers)} layers "
            f"could be repaired from {model_id}. Refusing to extract features from a "
            "partially initialised encoder (spec 7.1).")
    return {"applied": True, "layers_repaired": repaired}


class VideoMAEEncoder(Encoder):
    name = "videomae_b"
    is_video = True
    views = ("mean",)

    def __init__(self, layers=DEFAULT_LAYERS, device: str | None = None,
                 model_id: str = MODEL_ID, n_frames: int | None = None):
        import torch
        from transformers import AutoImageProcessor, VideoMAEModel

        self.layers = tuple(int(v) for v in layers)
        self._torch = torch
        self.device = pick_device(device)

        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.model = VideoMAEModel.from_pretrained(model_id)
        self.bias_repair_ = repair_attention_biases(self.model, model_id)
        self.model.eval().to(self.device)
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

        config = self.model.config
        self.n_frames = int(n_frames or getattr(config, "num_frames", 16))
        self.tubelet = int(getattr(config, "tubelet_size", 2))
        if self.n_frames % self.tubelet:
            raise ValueError(f"n_frames={self.n_frames} must be a multiple of tubelet {self.tubelet}")

        self.image_mean = list(getattr(self.processor, "image_mean", (0.485, 0.456, 0.406)))
        self.image_std = list(getattr(self.processor, "image_std", (0.229, 0.224, 0.225)))
        size = getattr(self.processor, "size", None) or {}
        self.input_size = int(size.get("shortest_edge") or size.get("height") or 224)

    def _resample(self, clip_length: int) -> np.ndarray:
        """Map the checkpoint's frame count onto our captured frames.

        The corpus captures 10 frames per scene (spec 5.2) and this checkpoint
        expects 16. Nearest-index resampling duplicates frames rather than
        blending them: an interpolated frame shows an object in a place it never
        occupied, which for a probe about physical state is a fabricated
        observation, not a smoothing.
        """
        return np.clip(np.round(np.linspace(0, clip_length - 1, self.n_frames)), 0,
                       clip_length - 1).astype(int)

    def embed(self, frames: np.ndarray) -> dict[int, dict[str, np.ndarray]]:
        torch = self._torch
        frames = np.asarray(frames)
        if frames.ndim != 5:
            raise ValueError(f"{self.name} is a video encoder; expected (N, T, H, W, 3), got {frames.shape}")

        clip = frames[:, self._resample(frames.shape[1])]            # (N, n_frames, H, W, 3)
        n, t = clip.shape[:2]

        from .base import to_normalised_tensor
        flat = to_normalised_tensor(clip.reshape(n * t, *clip.shape[2:]),
                                    self.image_mean, self.image_std,
                                    size=self.input_size, device=self.device)
        batch = flat.view(n, t, 3, self.input_size, self.input_size)

        with torch.no_grad():
            outputs = self.model(pixel_values=batch, output_hidden_states=True)

        # hidden_states[0] is the embedding output; block i is at index i+1.
        hidden = outputs.hidden_states
        result: dict[int, dict[str, np.ndarray]] = {}
        for layer in self.layers:
            tokens = hidden[layer + 1]
            result[layer] = {"mean": tokens.mean(dim=1).float().cpu().numpy(), "patch": None}
        return result

    def close(self) -> None:
        self.model = None
        if self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()


@register("videomae_b")
def _factory(**kwargs) -> VideoMAEEncoder:
    return VideoMAEEncoder(**kwargs)
