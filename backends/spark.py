"""Spark-TTS backend (mlx-audio).

Lightweight zero-shot voice cloning, native MLX. Same ref_audio + ref_text
interface as Qwen3 — works directly with VoiceProfile.

Lazy-loaded: model download + load happens on first request, not at startup,
so default Qwen3 boot stays fast.
"""
from __future__ import annotations
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Tuple

import numpy as np
from huggingface_hub import snapshot_download
from mlx_audio.tts.utils import load_model

from backends.base import Backend

if TYPE_CHECKING:
    from engine import VoiceProfile

logger = logging.getLogger("sanmarcsoft-tts")

# 4-6bit variant is the smallest on mlx-community (~250-350MB resident).
# Override via env if you need a different quantization (bf16 / 6bit / 8bit).
MODEL_ID = os.getenv("TTS_SPARK_MODEL", "mlx-community/Spark-TTS-0.5B-4-6bit")


class SparkBackend(Backend):
    name = "spark"
    # Spark-TTS-0.5B uses 16 kHz native output.
    sample_rate = 16000

    def __init__(self):
        self.model = None
        self.model_id = MODEL_ID

    def bootstrap(self) -> bool:
        # Lazy bootstrap — defer the snapshot_download until first use.
        # Server startup stays fast; the cost is paid on the first /v1/audio/speech
        # request with backend=spark.
        return True

    def load(self) -> bool:
        try:
            logger.info(f"[spark] Downloading + loading MLX Model: {self.model_id}")
            snapshot_download(repo_id=self.model_id)
            self.model = load_model(self.model_id)
            return True
        except Exception as e:
            logger.error(f"[spark] Model load failed: {e}")
            return False

    def is_loaded(self) -> bool:
        return self.model is not None

    def generate(self, text: str, voice: "VoiceProfile") -> Tuple[np.ndarray, int]:
        # Spark accepts a filesystem path for ref_audio (its loader resamples
        # to 16kHz internally). The path here is the DF3-cleaned WAV when
        # cleaning is enabled, else the source ref WAV — same precedence as Qwen.
        # Cap max_tokens proportional to word count. Spark's default (3000) lets
        # the autoregressive generator run for 30+ seconds on short prompts when
        # EOS detection is weak (e.g. on some persona embeddings). Scale at
        # ~25 tokens/word with a floor of 50 and a hard ceiling from env.
        word_count = len(text.split())
        spark_max_tokens_ceiling = int(os.getenv("TTS_SPARK_MAX_TOKENS_CEILING", "1500"))
        max_tokens = min(spark_max_tokens_ceiling, max(50, word_count * 25))

        gen_kwargs = {
            "text": text,
            "ref_audio": Path(voice.wav_path),
            "ref_text": voice.ref_text if voice.use_icl else None,
            "temperature": float(os.getenv("TTS_SPARK_TEMPERATURE", "0.8")),
            "top_p": float(os.getenv("TTS_SPARK_TOP_P", "0.95")),
            "top_k": int(os.getenv("TTS_SPARK_TOP_K", "50")),
            "max_tokens": max_tokens,
        }

        logger.info(
            f"[spark] Inference: voice={voice.name}, words={word_count}, "
            f"max_tokens={max_tokens}, text='{text[:40]}...'"
        )
        results = list(self.model.generate(**gen_kwargs))
        if not results:
            raise RuntimeError("spark.generate yielded no results")
        audio_array = np.array(results[0].audio)
        return (audio_array, self.sample_rate)
