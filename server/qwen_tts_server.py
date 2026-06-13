# coding=utf-8
# SPDX-License-Identifier: Apache-2.0
#
# Phenom-earth fork addition: an OpenAI-compatible HTTP server around the
# upstream `qwen_tts` library so PAI (and any OpenAI-audio client) can reach a
# single Qwen3-TTS instance over HTTP.
#
# Upstream ships a Python library + a gradio demo, but no HTTP API. PAI's voice
# layer speaks the OpenAI "audio/speech" dialect (POST /v1/audio/speech), so this
# wrapper exposes exactly that and forwards to `Qwen3TTSModel.generate_custom_voice`.
#
# It is meant to run on the Mac Studio (bare metal, Apple Silicon / MPS). Each
# Code Server container reaches it across the container->host boundary
# (host.docker.internal). See docs/SERVER_SETUP.md.
"""Qwen3-TTS OpenAI-compatible speech server.

Endpoints:
    POST /v1/audio/speech   OpenAI-compatible text-to-speech (returns audio bytes)
    GET  /v1/voices         list the model's supported speakers + languages
    GET  /v1/models         minimal OpenAI-style model list
    GET  /healthz           liveness (process up)
    GET  /readyz            readiness (model loaded)

Configuration is entirely via environment variables (see CONFIG below) so the
same image/script runs unchanged on the Mac Studio and in CI.
"""
from __future__ import annotations

import io
import json
import logging
import os
from contextlib import asynccontextmanager
from typing import Optional

logger = logging.getLogger("qwen_tts_server")


# --------------------------------------------------------------------------- #
# Configuration (env-driven)
# --------------------------------------------------------------------------- #
class Config:
    # HF repo id OR a local checkpoint directory downloaded ahead of time.
    model = os.environ.get("QWEN_TTS_MODEL", "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice")
    host = os.environ.get("QWEN_TTS_HOST", "0.0.0.0")  # 0.0.0.0 so containers can reach it
    port = int(os.environ.get("QWEN_TTS_PORT", "8880"))
    device = os.environ.get("QWEN_TTS_DEVICE", "auto")  # "mps" on Apple Silicon, "cuda", "cpu"
    dtype = os.environ.get("QWEN_TTS_DTYPE", "bfloat16")
    default_speaker = os.environ.get("QWEN_TTS_DEFAULT_SPEAKER", "")  # "" => first supported
    default_language = os.environ.get("QWEN_TTS_DEFAULT_LANGUAGE", "Auto")
    # Optional bearer token. If set, every /v1/* request must send
    # `Authorization: Bearer <token>`. Strongly recommended when host is 0.0.0.0.
    api_key = os.environ.get("QWEN_TTS_API_KEY", "")
    # JSON map of OpenAI/PAI voice name -> Qwen speaker, e.g. {"narrator":"Ethan"}.
    voice_aliases = json.loads(os.environ.get("QWEN_TTS_VOICE_ALIASES", "{}"))


CONFIG = Config()


# --------------------------------------------------------------------------- #
# Model holder (loaded once at startup)
# --------------------------------------------------------------------------- #
class _State:
    model = None
    speakers: list[str] = []
    languages: list[str] = []
    sample_rate: int = 24000


STATE = _State()


def _resolve_dtype(name: str):
    import torch

    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
        "auto": "auto",
    }.get(name.lower(), torch.bfloat16)


def _load_model() -> None:
    """Load the Qwen3-TTS CustomVoice model once. Raises on failure so the
    process exits loudly rather than serving 500s forever."""
    from qwen_tts import Qwen3TTSModel

    logger.info("Loading Qwen3-TTS model=%s device=%s dtype=%s", CONFIG.model, CONFIG.device, CONFIG.dtype)
    model = Qwen3TTSModel.from_pretrained(
        CONFIG.model,
        device_map=CONFIG.device,
        dtype=_resolve_dtype(CONFIG.dtype),
        attn_implementation=None,  # flash-attn-2 is unavailable on Apple Silicon
    )
    STATE.model = model
    try:
        STATE.speakers = list(model.get_supported_speakers() or [])
    except Exception:  # pragma: no cover - model variant may not expose it
        STATE.speakers = []
    try:
        STATE.languages = list(model.get_supported_languages() or [])
    except Exception:  # pragma: no cover
        STATE.languages = []
    logger.info("Model ready. speakers=%s languages=%s", STATE.speakers, STATE.languages)


def _pick_speaker(requested: Optional[str]) -> str:
    """Map an OpenAI/PAI `voice` to a real Qwen speaker.

    Resolution order: explicit alias -> exact supported speaker (case-insensitive)
    -> configured default -> first supported speaker. Never 400s on an unknown
    voice so generic OpenAI clients ("alloy", "nova", ...) still get audio.
    """
    name = (requested or "").strip()
    if name and name in CONFIG.voice_aliases:
        return CONFIG.voice_aliases[name]
    if name and STATE.speakers:
        for s in STATE.speakers:
            if s.lower() == name.lower():
                return s
    if CONFIG.default_speaker:
        return CONFIG.default_speaker
    if STATE.speakers:
        return STATE.speakers[0]
    return name or "default"


def _encode_audio(wav, sample_rate: int, fmt: str) -> tuple[bytes, str]:
    """Encode a float waveform to the requested container. WAV/PCM are always
    available; flac/ogg/opus go through libsndfile; mp3 falls back to wav."""
    import numpy as np
    import soundfile as sf

    arr = np.asarray(wav, dtype=np.float32)
    fmt = (fmt or "wav").lower()
    if fmt == "pcm":  # raw 16-bit little-endian, OpenAI "pcm"
        pcm16 = np.clip(arr, -1.0, 1.0)
        return (pcm16 * 32767.0).astype("<i2").tobytes(), "audio/pcm"

    sf_format = {"wav": "WAV", "flac": "FLAC", "ogg": "OGG", "opus": "OGG", "mp3": "WAV"}.get(fmt)
    if sf_format is None:
        raise ValueError(f"unsupported response_format: {fmt}")
    if fmt == "mp3":
        logger.warning("mp3 requested but not supported by libsndfile build; returning wav")
        fmt = "wav"
    buf = io.BytesIO()
    subtype = "OPUS" if fmt == "opus" else None
    sf.write(buf, arr, sample_rate, format=sf_format, subtype=subtype)
    media = {"wav": "audio/wav", "flac": "audio/flac", "ogg": "audio/ogg", "opus": "audio/ogg"}[fmt]
    return buf.getvalue(), media


# --------------------------------------------------------------------------- #
# FastAPI app
# --------------------------------------------------------------------------- #
def create_app():
    from fastapi import Depends, FastAPI, Header, HTTPException
    from fastapi.responses import JSONResponse, Response
    from pydantic import BaseModel

    @asynccontextmanager
    async def lifespan(app):  # noqa: ANN001
        _load_model()
        yield

    app = FastAPI(title="Qwen3-TTS Server (Phenom-earth)", version="1.0.0", lifespan=lifespan)

    def require_auth(authorization: str | None = Header(default=None)) -> None:
        if not CONFIG.api_key:
            return
        expected = f"Bearer {CONFIG.api_key}"
        if authorization != expected:
            raise HTTPException(status_code=401, detail="invalid or missing bearer token")

    class SpeechRequest(BaseModel):
        model: str | None = None
        input: str
        voice: str | None = None
        response_format: str | None = "wav"
        speed: float | None = 1.0
        instructions: str | None = None  # OpenAI 'instructions' -> Qwen 'instruct'
        language: str | None = None  # Qwen extension

    @app.get("/healthz")
    async def healthz():
        return {"status": "ok"}

    @app.get("/readyz")
    async def readyz():
        if STATE.model is None:
            return JSONResponse(status_code=503, content={"status": "loading"})
        return {"status": "ready", "model": CONFIG.model}

    @app.get("/v1/models")
    async def list_models(_: None = Depends(require_auth)):
        return {"object": "list", "data": [{"id": CONFIG.model, "object": "model", "owned_by": "qwen"}]}

    @app.get("/v1/voices")
    async def list_voices(_: None = Depends(require_auth)):
        return {"speakers": STATE.speakers, "languages": STATE.languages, "aliases": CONFIG.voice_aliases}

    @app.post("/v1/audio/speech")
    async def create_speech(req: SpeechRequest, _: None = Depends(require_auth)):
        if STATE.model is None:
            raise HTTPException(status_code=503, detail="model still loading")
        text = (req.input or "").strip()
        if not text:
            raise HTTPException(status_code=400, detail="input text is required")
        speaker = _pick_speaker(req.voice)
        language = req.language or CONFIG.default_language
        try:
            wavs, sr = STATE.model.generate_custom_voice(
                text=text,
                speaker=speaker,
                language=language,
                instruct=(req.instructions or None),
            )
        except Exception as exc:  # surface model errors as 400, not opaque 500
            logger.exception("generation failed")
            raise HTTPException(status_code=400, detail=f"generation failed: {exc}") from exc
        wav = wavs[0] if isinstance(wavs, (list, tuple)) else wavs
        audio, media_type = _encode_audio(wav, int(sr), req.response_format or "wav")
        return Response(content=audio, media_type=media_type)

    return app


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    import uvicorn

    if CONFIG.host not in ("127.0.0.1", "localhost", "::1") and not CONFIG.api_key:
        logger.warning(
            "Binding %s with no QWEN_TTS_API_KEY set; the server is reachable on the "
            "local network without auth. Set QWEN_TTS_API_KEY before exposing it.",
            CONFIG.host,
        )
    uvicorn.run(create_app(), host=CONFIG.host, port=CONFIG.port, log_level="info")


if __name__ == "__main__":
    main()
