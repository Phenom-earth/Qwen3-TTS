"""Per-voice ffmpeg post-processing. Encodes raw float32 PCM to MP3 or WAV
with optional per-voice -af chains. Lifted as-is from the pre-refactor server.
"""
from __future__ import annotations
import io
import logging
import os
import subprocess
from typing import Tuple

import numpy as np
import soundfile as sf

logger = logging.getLogger("sanmarcsoft-tts")

MP3_BITRATE = os.getenv("TTS_MP3_BITRATE", "128k")

VOICE_FX: dict[str, str] = {
    # Narrator: +30% gain, AM radio bandpass (200-4000Hz), DJ compression
    "narrator": "volume=1.6,highpass=f=120,lowpass=f=8000,acompressor=threshold=-18dB:ratio=3:attack=5:release=50",
    # Master Control: no FX (placeholder)
    "master-control": "",
    # 007: +100% gain — baseline too quiet for M
    "007": "volume=2.0",
}


def audio_to_response_bytes(
    audio_array: np.ndarray, sample_rate: int, response_format: str, voice_id: str = ""
) -> Tuple[bytes, str]:
    """Encode float32 PCM → WAV bytes → ffmpeg (optional FX + format conversion)."""
    wav_io = io.BytesIO()
    sf.write(wav_io, audio_array, sample_rate, format="WAV")
    wav_bytes = wav_io.getvalue()
    return _ffmpeg_convert(wav_bytes, response_format, voice_id, sample_rate)


def _ffmpeg_convert(
    wav_bytes: bytes, response_format: str, voice_id: str, sample_rate: int
) -> Tuple[bytes, str]:
    fx = VOICE_FX.get(voice_id, "")
    af_args = ["-af", fx] if fx else []

    if response_format == "wav":
        cmd = ["ffmpeg", "-y", "-loglevel", "error",
               "-i", "pipe:0"] + af_args + [
               "-ar", str(sample_rate), "-ac", "1", "-sample_fmt", "s16",
               "-f", "wav", "pipe:1"]
        media_type = "audio/wav"
    else:
        cmd = ["ffmpeg", "-y", "-loglevel", "error",
               "-i", "pipe:0"] + af_args + [
               "-codec:a", "libmp3lame", "-b:a", MP3_BITRATE,
               "-f", "mp3", "pipe:1"]
        media_type = "audio/mpeg"

    try:
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = proc.communicate(input=wav_bytes, timeout=30)
        if proc.returncode != 0:
            logger.error(f"FFmpeg error (rc={proc.returncode}): {stderr.decode()[:200]}")
            if response_format == "wav":
                return (wav_bytes, "audio/wav")
            raise subprocess.CalledProcessError(proc.returncode, cmd)
        return (stdout, media_type)
    except subprocess.TimeoutExpired:
        proc.kill()
        logger.error("FFmpeg timed out after 30s")
        if response_format == "wav":
            return (wav_bytes, "audio/wav")
        raise
