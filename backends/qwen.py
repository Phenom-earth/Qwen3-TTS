"""Qwen3-TTS backend (mlx-audio). Behavior preserved from pre-refactor server."""
from __future__ import annotations
import logging
import os
from typing import TYPE_CHECKING, Tuple

import numpy as np
import mlx.core as mx
import soundfile as sf
from huggingface_hub import snapshot_download
from mlx_audio.tts.utils import load_model

from backends.base import Backend

if TYPE_CHECKING:
    from engine import VoiceProfile

logger = logging.getLogger("sanmarcsoft-tts")

MODEL_ID = os.getenv("TTS_QWEN_MODEL", "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-4bit")


class QwenBackend(Backend):
    name = "qwen"
    sample_rate = 24000

    def __init__(self):
        self.model = None
        self.model_id = MODEL_ID

    def bootstrap(self) -> bool:
        try:
            snapshot_download(repo_id=self.model_id)
            return True
        except Exception as e:
            logger.error(f"[qwen] Bootstrap failure: {e}")
            return False

    def load(self) -> bool:
        try:
            logger.info(f"[qwen] Loading MLX Model: {self.model_id}")
            self.model = load_model(self.model_id)
            return True
        except Exception as e:
            logger.error(f"[qwen] Model load failed: {e}")
            return False

    def is_loaded(self) -> bool:
        return self.model is not None

    def generate(self, text: str, voice: "VoiceProfile") -> Tuple[np.ndarray, int]:
        # Use preloaded MLX audio array (cached in memory), fall back to disk read.
        if voice.cached_mlx_audio is not None:
            mlx_audio_data = voice.cached_mlx_audio
        else:
            logger.warning(f"[qwen] Voice '{voice.name}': cache miss, reading ref audio from disk")
            audio_data, _sr = sf.read(voice.wav_path)
            if len(audio_data.shape) > 1:
                audio_data = np.mean(audio_data, axis=1)
            audio_data = audio_data.astype(np.float32)
            mlx_audio_data = mx.array(audio_data)

        # When use_icl=False, pass ref_text=None so the model uses ONLY the
        # x-vector speaker embedding and does NOT inject reference dialogue.
        effective_ref_text = voice.ref_text if voice.use_icl else None
        gen_kwargs = {
            "text": text,
            "ref_audio": mlx_audio_data,
            "ref_text": effective_ref_text,
            "task_type": "Base",
        }
        if not voice.use_icl:
            # Without ICL the model doesn't self-terminate as reliably; cap tokens.
            word_count = len(text.split())
            smart_max = min(4096, max(75, int(word_count * 1.3 * 6)))
            gen_kwargs["max_tokens"] = smart_max
            logger.info(f"[qwen] x-vector-only mode: max_tokens={smart_max} for {word_count} words")

        logger.info(f"[qwen] Inference: voice={voice.name}, text='{text[:40]}...'")
        results = list(self.model.generate(**gen_kwargs))

        if voice.cached_embedding is None:
            emb = getattr(results[0], "speaker_embedding", None)
            if emb is not None:
                voice.save_embedding(emb)
                logger.info(f"[qwen] Saved speaker embedding for voice: {voice.name}")

        audio_array = np.array(results[0].audio)
        return (audio_array, self.sample_rate)
