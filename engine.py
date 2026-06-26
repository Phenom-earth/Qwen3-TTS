"""Voice profiles + TTS engine orchestration. Backend-agnostic — dispatches
generation through a registered backend, then runs per-voice ffmpeg post-FX.
"""
from __future__ import annotations
import asyncio
import logging
import os
from typing import Dict, Optional, Tuple

import numpy as np
import mlx.core as mx
import soundfile as sf
from fastapi import HTTPException

from backends import all_backends, get as get_backend
from backends.base import Backend
from effects import audio_to_response_bytes

logger = logging.getLogger("sanmarcsoft-tts")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_VOICE = os.getenv("TTS_DEFAULT_VOICE", "q")
DEFAULT_BACKEND = os.getenv("TTS_DEFAULT_BACKEND", "qwen")


class VoiceProfile:
    """A voice profile backed by {name}_ref.wav + {name}_ref.txt files.
    Holds reference audio + metadata; backend-agnostic (any backend that
    accepts ref-audio cloning consumes this directly).

    Refs are loaded raw — no cleaning pipeline. DF3 was tried and removed
    after it stripped the vintage tape character that defines period voice
    sources (Rodriguez trailers, Grover commercials). Backends consume the
    source WAV directly.
    """

    def __init__(self, name: str, wav_path: str, txt_path: str):
        self.name = name
        self.source_wav_path = wav_path  # kept for symmetry; equals wav_path now
        self.wav_path = wav_path
        self.txt_path = txt_path
        self.ref_text = ""
        self.use_icl = True
        self.cached_embedding = None
        self.cached_mlx_audio = None
        self.embedding_cache_path = os.path.join(SCRIPT_DIR, f".{name}_speaker_embedding.npy")
        self.mtime_cache_path = os.path.join(SCRIPT_DIR, f".{name}_mtime.txt")
        self.no_icl_path = os.path.join(SCRIPT_DIR, f"{name}_ref.no_icl")

    def load_ref_text(self):
        if os.path.exists(self.txt_path):
            with open(self.txt_path, "r") as f:
                self.ref_text = f.read().strip()

    def preload_audio(self):
        audio_data, _sr = sf.read(self.wav_path)
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)
        audio_data = audio_data.astype(np.float32)
        self.cached_mlx_audio = mx.array(audio_data)
        logger.info(f"Voice '{self.name}': ref audio preloaded ({len(audio_data)} samples)")

    def load_icl_mode(self):
        self.use_icl = not os.path.exists(self.no_icl_path)

    def get_mtime(self) -> float:
        # Use the SOURCE WAV's mtime so cache validity tracks the user's
        # edits to {name}_ref.wav, not our cleaning cache file timestamps.
        mtime = os.path.getmtime(self.source_wav_path)
        if os.path.exists(self.txt_path):
            mtime = max(mtime, os.path.getmtime(self.txt_path))
        return mtime

    def is_cache_valid(self) -> bool:
        if not os.path.exists(self.embedding_cache_path) or not os.path.exists(self.mtime_cache_path):
            return False
        try:
            with open(self.mtime_cache_path, "r") as f:
                saved = float(f.read().strip())
            return abs(self.get_mtime() - saved) < 1e-4
        except Exception:
            return False

    def load_cached_embedding(self) -> bool:
        if self.is_cache_valid():
            try:
                self.cached_embedding = np.load(self.embedding_cache_path)
                return True
            except Exception:
                pass
        return False

    def save_embedding(self, embedding):
        self.cached_embedding = embedding
        np.save(self.embedding_cache_path, embedding)
        with open(self.mtime_cache_path, "w") as f:
            f.write(str(self.get_mtime()))

    def clear_cache(self):
        self.cached_embedding = None
        self.cached_mlx_audio = None
        for fp in [self.embedding_cache_path, self.mtime_cache_path]:
            if os.path.exists(fp):
                os.remove(fp)

    def to_dict(self) -> dict:
        return {
            "id": self.name,
            "name": self.name.title(),
            "description": f"Voice clone from {self.name}_ref.wav",
            "model": "qwen3-tts",
            "sample_rate": 24000,
            "language": "en",
            "ref_audio": f"{self.name}_ref.wav",
            "ref_text_preview": self.ref_text[:80] + "..." if len(self.ref_text) > 80 else self.ref_text,
            "embedding_cached": self.cached_embedding is not None,
            "audio_preloaded": self.cached_mlx_audio is not None,
            "use_icl": self.use_icl,
            "reclone_endpoint": f"/v1/voices/{self.name}/reclone",
        }


class TTSEngine:
    """Multi-voice, multi-backend TTS dispatcher."""

    def __init__(self):
        self.voices: Dict[str, VoiceProfile] = {}
        self.sem = asyncio.Semaphore(1)

    def discover_voices(self):
        discovered = {}
        for fname in os.listdir(SCRIPT_DIR):
            if fname.endswith("_ref.wav"):
                name = fname.replace("_ref.wav", "")
                wav_path = os.path.join(SCRIPT_DIR, fname)
                txt_path = os.path.join(SCRIPT_DIR, f"{name}_ref.txt")
                profile = VoiceProfile(name, wav_path, txt_path)
                profile.load_ref_text()
                profile.load_icl_mode()
                profile.load_cached_embedding()
                profile.preload_audio()
                discovered[name] = profile
                logger.info(
                    f"Voice discovered: {name} "
                    f"(embedding_cached={profile.cached_embedding is not None}, "
                    f"audio_preloaded={profile.cached_mlx_audio is not None})"
                )
        self.voices = discovered
        logger.info(f"Total voices: {len(self.voices)} ({', '.join(self.voices.keys())})")

    def bootstrap(self) -> bool:
        """Discover voices and bootstrap every registered backend (snapshot_download).
        Returns True if at least the default backend bootstraps successfully.
        """
        logger.info("Starting TTS Bootstrap...")
        self.discover_voices()
        if not self.voices:
            logger.error("No voice reference files found (*_ref.wav)")
            return False
        if DEFAULT_VOICE not in self.voices:
            logger.warning(
                f"Default voice '{DEFAULT_VOICE}' not found. Available: {list(self.voices.keys())}"
            )
        any_ok = False
        for be in all_backends().values():
            if be.bootstrap():
                any_ok = True
            else:
                logger.warning(f"Backend bootstrap failed (non-fatal): {be.name}")
        return any_ok

    def load_default(self) -> bool:
        """Eager-load the default backend; others lazy-load on first use."""
        be = get_backend(DEFAULT_BACKEND)
        return be.load()

    def get_voice(self, name: str) -> VoiceProfile:
        if name not in self.voices:
            available = ", ".join(self.voices.keys())
            raise HTTPException(status_code=400, detail=f"Unknown voice: {name}. Available: {available}")
        return self.voices[name]

    def get_backend_or_load(self, name: str) -> Backend:
        be = get_backend(name)
        if not be.is_loaded():
            logger.info(f"Lazy-loading backend: {name}")
            if not be.load():
                raise HTTPException(status_code=500, detail=f"Failed to load backend: {name}")
        return be

    async def generate_audio(
        self,
        text: str,
        voice_name: str = DEFAULT_VOICE,
        response_format: str = "mp3",
        backend_name: Optional[str] = None,
    ) -> Tuple[bytes, str]:
        chosen = backend_name or DEFAULT_BACKEND
        return await asyncio.to_thread(self._sync_inference, text, voice_name, response_format, chosen)

    def _sync_inference(
        self, text: str, voice_name: str, response_format: str, backend_name: str
    ) -> Tuple[bytes, str]:
        voice = self.get_voice(voice_name)
        backend = self.get_backend_or_load(backend_name)
        logger.info(
            f"Inference: backend={backend_name}, voice={voice_name}, "
            f"text='{text[:40]}...', format={response_format}"
        )
        audio_array, sample_rate = backend.generate(text, voice)
        return audio_to_response_bytes(audio_array, sample_rate, response_format, voice_name)
