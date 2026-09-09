"""Frozen encoders (spec 7.1).

Concrete encoders are imported lazily via :func:`physground.encoders.get_encoder`
so that a CPU-only corpus-generation container never imports torch and a run
touching only DINOv2 never imports transformers.
"""

from .base import Encoder, available_encoders, get_encoder, register

__all__ = ["Encoder", "available_encoders", "get_encoder", "register"]
