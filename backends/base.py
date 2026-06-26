"""Backend ABC. A backend loads a specific MLX-audio model and produces
raw audio (numpy float32) at its native sample rate from text + voice profile.

The engine is backend-agnostic: it owns voices, dispatches generation
through a registered backend, then runs per-voice ffmpeg post-processing.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Tuple
import numpy as np

if TYPE_CHECKING:
    from engine import VoiceProfile


class Backend(ABC):
    name: str = "?"
    sample_rate: int = 0

    @abstractmethod
    def bootstrap(self) -> bool:
        """Pre-flight: snapshot_download weights, validate env. Idempotent."""
        ...

    @abstractmethod
    def load(self) -> bool:
        """Materialize the model into memory. May be eager (startup) or lazy."""
        ...

    @abstractmethod
    def is_loaded(self) -> bool:
        ...

    @abstractmethod
    def generate(self, text: str, voice: "VoiceProfile") -> Tuple[np.ndarray, int]:
        """Return (mono float32 ndarray, sample_rate)."""
        ...

    def unload(self) -> None:
        """Free RAM. Default no-op."""
        pass
