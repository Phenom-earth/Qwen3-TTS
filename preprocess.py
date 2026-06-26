"""Reference-audio cleaning pipeline (DeepFilterNet3).

- preprocess_reference(src_wav) — return path to a cleaned copy of the WAV,
  cached on disk and keyed by (source size, mtime_ns) so the heavy DF3
  inference runs at most once per ref WAV revision.
- Lazy-loads DF3 on first call; subsequent calls reuse the in-process state.
- Kill-switch via env TTS_ENABLE_DENOISE_REF=0 — when off, returns the
  original WAV path untouched (no model load, no inference).

Why the import dance: DeepFilterNet 0.5.6's df/io.py imports
`torchaudio.backend.common.AudioMetaData`, a module path torchaudio dropped
in 2.4+. There are no torch wheels < 2.9 for Python 3.14, so we can't
downgrade torchaudio. Workaround: install runtime stubs for the removed
torchaudio paths BEFORE df is imported, then bypass df.io entirely by
doing our own soundfile-based IO. df.enhance + init_df work fine; only the
io shim breaks.
"""
from __future__ import annotations
import hashlib
import logging
import os
import sys
import threading
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import soundfile as sf

logger = logging.getLogger("sanmarcsoft-tts")

SCRIPT_DIR = Path(__file__).resolve().parent
CACHE_DIR = SCRIPT_DIR / ".cache" / "cleaned"
ENABLE_DENOISE_REF = os.getenv("TTS_ENABLE_DENOISE_REF", "1") not in ("0", "false", "False", "")

# Lazy-loaded DF3 state — protected by a lock so concurrent first-callers
# don't race the model load.
_df_lock = threading.Lock()
_df_model = None
_df_state = None


def _install_torchaudio_stubs() -> None:
    """Patch `torchaudio.backend.common.AudioMetaData` into existence so that
    `df/io.py` (referenced transitively by df.enhance) imports without error.
    df.io.load_audio / save_audio are unused — preprocess.py does its own IO.
    """
    import torchaudio  # noqa: F401  (touched only for module identity)

    class _StubAudioMetaData:
        """Stand-in for the dropped torchaudio.backend.common.AudioMetaData."""
        sample_rate: int = 0
        num_frames: int = 0
        num_channels: int = 0
        bits_per_sample: int = 0
        encoding: str = ""

    if "torchaudio.backend.common" not in sys.modules:
        class _BackendCommon:
            AudioMetaData = _StubAudioMetaData

        class _Backend:
            common = _BackendCommon

        sys.modules["torchaudio.backend"] = _Backend
        sys.modules["torchaudio.backend.common"] = _BackendCommon
        # Also poke the attribute path on torchaudio so `from torchaudio.backend.common`
        # resolves through the live module object.
        torchaudio.backend = _Backend  # type: ignore[attr-defined]


def _ensure_df_loaded() -> Tuple[object, object]:
    """Idempotent DF3 lazy-load. Returns (model, df_state)."""
    global _df_model, _df_state
    if _df_model is not None and _df_state is not None:
        return _df_model, _df_state
    with _df_lock:
        if _df_model is not None and _df_state is not None:
            return _df_model, _df_state
        logger.info("[preprocess] Loading DeepFilterNet3 (first ref-clean call)")
        _install_torchaudio_stubs()
        # Import only after stubs are live.
        from df.enhance import enhance, init_df  # noqa: F401  (enhance used later)
        m, s, _ = init_df()
        _df_model = m
        _df_state = s
        logger.info(f"[preprocess] DF3 ready (model sr={s.sr()} Hz)")
        return _df_model, _df_state


def _cache_key(src: Path) -> str:
    """Stable key tied to (source path's name, size, mtime_ns)."""
    st = src.stat()
    raw = f"{src.name}|{st.st_size}|{st.st_mtime_ns}".encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:12]


def _cache_path(src: Path) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"{src.stem}_{_cache_key(src)}.wav"


def preprocess_reference(src_wav: Path | str) -> Path:
    """Return a path to a denoised copy of `src_wav`, building it once and
    reusing on subsequent calls.

    Falls back to returning the original path unchanged when:
      - TTS_ENABLE_DENOISE_REF is off
      - DF3 import / inference fails for any reason

    Output WAV preserves the source sample rate (DF3 runs at 48kHz internally
    and we resample back so downstream TTS models see the same SR they saw
    before).
    """
    src = Path(src_wav).resolve()
    if not ENABLE_DENOISE_REF:
        return src

    cache_path = _cache_path(src)
    if cache_path.exists():
        logger.info(f"[preprocess] cache hit: {src.name} → {cache_path.name}")
        return cache_path

    try:
        model, df_state = _ensure_df_loaded()
    except Exception as e:
        logger.warning(f"[preprocess] DF3 unavailable, using source unchanged: {e}")
        return src

    try:
        import torch  # local import — only needed if we actually run DF3
        from df.enhance import enhance

        df_sr = df_state.sr()
        # Read source via soundfile (avoids torchaudio.load which now requires torchcodec).
        audio_np, src_sr = sf.read(str(src), dtype="float32")
        if audio_np.ndim > 1:
            audio_np = audio_np.mean(axis=1)  # downmix to mono

        # DF3 wants a torch tensor of shape [channels, samples] at its native sr.
        audio_t = torch.from_numpy(audio_np).unsqueeze(0)
        if src_sr != df_sr:
            audio_t = torch.nn.functional.interpolate(
                audio_t.unsqueeze(0),
                scale_factor=df_sr / src_sr,
                mode="linear",
                align_corners=False,
            ).squeeze(0)

        enhanced_t = enhance(model, df_state, audio_t)

        # Resample back to source sr so downstream TTS doesn't see a sr change.
        if src_sr != df_sr:
            enhanced_t = torch.nn.functional.interpolate(
                enhanced_t.unsqueeze(0),
                scale_factor=src_sr / df_sr,
                mode="linear",
                align_corners=False,
            ).squeeze(0)

        cleaned_np = enhanced_t.squeeze(0).cpu().numpy().astype(np.float32)
        # Atomic write: write to .tmp then rename so a partial file is never visible.
        tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
        sf.write(str(tmp_path), cleaned_np, src_sr, format="WAV", subtype="PCM_16")
        tmp_path.replace(cache_path)
        logger.info(f"[preprocess] cleaned: {src.name} ({len(audio_np)} samples @ {src_sr}Hz) → {cache_path.name}")
        return cache_path
    except Exception as e:
        logger.warning(f"[preprocess] DF3 inference failed, using source unchanged: {e}")
        return src


def clear_cache(name: Optional[str] = None) -> int:
    """Remove cached cleaned WAVs. If `name` is given, only entries whose
    filename starts with `{name}_` are removed. Returns count removed.
    """
    if not CACHE_DIR.exists():
        return 0
    removed = 0
    prefix = f"{name}_" if name else None
    for f in CACHE_DIR.glob("*.wav"):
        if prefix is None or f.name.startswith(prefix):
            try:
                f.unlink()
                removed += 1
            except OSError:
                pass
    return removed
