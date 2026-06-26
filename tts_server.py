"""SanMarcSoft TTS Server.

Refactored into modules:
  - engine.py        TTSEngine, VoiceProfile (backend-agnostic)
  - backends/        Backend ABC + concrete backends (qwen, ...)
  - effects.py       Per-voice ffmpeg post-processing

Commit 1: refactor only — API surface, response shapes, port/path unchanged.
Commit 2 will add reference-audio cleaning. Commit 3 will add the Spark backend
+ a backend selector.
"""
import logging
import os
from contextlib import asynccontextmanager
from typing import Optional

import mlx.core as mx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, ORJSONResponse
from pydantic import BaseModel

from engine import TTSEngine, DEFAULT_VOICE, DEFAULT_BACKEND
from backends import get as get_backend, all_backends
from backends.qwen import MODEL_ID  # surface-compat: /health still reports this

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("sanmarcsoft-tts")

engine = TTSEngine()


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not engine.bootstrap() or not engine.load_default():
        logger.critical("Startup failed. Check paths and venv.")
        os._exit(1)
    logger.info(
        f"SanMarcSoft TTS is live - {len(engine.voices)} voice(s): "
        f"{', '.join(engine.voices.keys())}"
    )

    # Pre-warm: run a short inference per voice to populate model caches.
    for voice_name in engine.voices:
        try:
            logger.info(f"Pre-warming voice: {voice_name}")
            await engine.generate_audio("Hello.", voice_name, "wav")
            logger.info(f"Pre-warm complete: {voice_name}")
        except Exception as e:
            logger.warning(f"Pre-warm failed for {voice_name} (non-fatal): {e}")

    logger.info("All voices pre-warmed and ready")
    yield
    mx.clear_cache()


app = FastAPI(
    title="SanMarcSoft TTS Server",
    description="Qwen3-TTS multi-voice server with OpenAI-compatible API",
    version="5.1.0",
    lifespan=lifespan,
    default_response_class=ORJSONResponse,
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class TTSRequest(BaseModel):
    input: str
    model: str = "qwen3-tts"
    voice: str = DEFAULT_VOICE
    response_format: str = "mp3"
    speed: float = 1.0
    backend: Optional[str] = None  # qwen | spark; falls back to ?backend= or DEFAULT_BACKEND


@app.post("/v1/audio/speech")
async def speech_endpoint(request: TTSRequest, backend: Optional[str] = None):
    """OpenAI-compatible text-to-speech endpoint. Backend selectable via
    JSON body `backend` field OR `?backend=qwen|spark` query param.
    Default backend is qwen — existing clients keep working unchanged.
    """
    if not request.input.strip():
        raise HTTPException(status_code=400, detail="Text is empty")
    fmt = request.response_format.lower()
    if fmt not in ("mp3", "wav"):
        raise HTTPException(status_code=400, detail=f"Unsupported format: {fmt}. Use mp3 or wav.")

    chosen_backend = request.backend or backend or DEFAULT_BACKEND
    available = list(all_backends().keys())
    if chosen_backend not in available:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown backend: {chosen_backend}. Available: {available}",
        )

    async with engine.sem:
        try:
            audio_bytes, media_type = await engine.generate_audio(
                request.input, request.voice, fmt, chosen_backend
            )
            return Response(content=audio_bytes, media_type=media_type)
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Inference Error ({chosen_backend}): {e}")
            raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/voices/{voice_name}/reclone")
async def reclone_voice(voice_name: str):
    """Clear cached embedding for a voice, forcing re-extraction on next call."""
    voice = engine.get_voice(voice_name)
    voice.clear_cache()
    return {"status": "success", "voice": voice_name, "message": "Cache cleared. Next request will re-clone."}


@app.post("/v1/voices/reload")
async def reload_voices():
    """Rescan directory for new voice reference files."""
    engine.discover_voices()
    return {
        "status": "success",
        "voices": [v.to_dict() for v in engine.voices.values()],
    }


@app.get("/v1/models")
async def list_models():
    """OpenAI-compatible model listing. Advertises every registered backend."""
    descriptions = {
        "qwen": "Qwen3-TTS 0.6B Base (4-bit MLX) with multi-voice support",
        "spark": "Spark-TTS 0.5B (4-6 bit MLX), zero-shot voice cloning",
    }
    return {
        "object": "list",
        "data": [
            {
                "id": be.name if be.name != "qwen" else "qwen3-tts",  # legacy alias
                "object": "model",
                "created": 1700000000,
                "owned_by": "sanmarcsoft",
                "description": descriptions.get(be.name, f"{be.name} TTS backend"),
                "loaded": be.is_loaded(),
                "capabilities": {
                    "audio": {
                        "output_formats": ["mp3", "wav"],
                        "sample_rate": be.sample_rate,
                        "voices": list(engine.voices.keys()),
                    }
                },
            }
            for be in all_backends().values()
        ],
    }


@app.get("/v1/voices")
async def list_voices():
    return {"voices": [v.to_dict() for v in engine.voices.values()]}


@app.get("/health")
async def health_check():
    return {
        "status": "ok" if get_backend(DEFAULT_BACKEND).is_loaded() else "loading",
        "model": MODEL_ID,  # legacy field — name of the default Qwen3 model
        "default_backend": DEFAULT_BACKEND,
        "backends": {
            be.name: {"loaded": be.is_loaded(), "sample_rate": be.sample_rate}
            for be in all_backends().values()
        },
        "voices": list(engine.voices.keys()),
        "default_voice": DEFAULT_VOICE,
        "supported_formats": ["mp3", "wav"],
        "sample_rate": 24000,
    }


@app.get("/")
async def root():
    return {
        "service": "SanMarcSoft TTS (Qwen3-TTS)",
        "version": "5.1.0",
        "model": MODEL_ID,
        "voices": list(engine.voices.keys()),
        "default_voice": DEFAULT_VOICE,
        "endpoints": {
            "speech": "POST /v1/audio/speech {input, voice?, response_format?, model?, backend?} or ?backend=qwen|spark",
            "models": "GET /v1/models",
            "voices": "GET /v1/voices",
            "reclone": "POST /v1/voices/{name}/reclone",
            "reload": "POST /v1/voices/reload",
            "health": "GET /health",
            "docs": "GET /docs",
        },
        "openai_compatible": True,
        "note": "Drop {name}_ref.wav + {name}_ref.txt in server dir, then POST /v1/voices/reload",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8880)
