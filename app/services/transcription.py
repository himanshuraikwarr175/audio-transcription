"""Speech-to-text with OpenAI Whisper.

The model is loaded lazily, once per worker process, and reused for every task
that process handles. Whisper is imported inside functions so the API process
(and the tests) never need torch or the model weights.
"""

import logging
import threading
from pathlib import Path
from typing import Any

from app.core.config import settings

logger = logging.getLogger(__name__)


class InvalidAudioError(Exception):
    """Permanent failure: the file cannot be decoded as audio. Never retried."""


class TransientTranscriptionError(Exception):
    """Temporary failure that is worth retrying."""


_model = None
_model_lock = threading.Lock()


def get_model():
    """Return the process-wide Whisper model, loading it on first use."""
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                import whisper

                logger.info("Loading Whisper model %r on %s", settings.whisper_model, settings.whisper_device)
                _model = whisper.load_model(settings.whisper_model, device=settings.whisper_device)
                logger.info("Whisper model loaded")
    return _model


def transcribe_file(path: str | Path) -> dict[str, Any]:
    """Transcribe an audio file.

    Returns {"text": str, "language": str | None, "segments": [{"start", "end", "text"}]}.
    Segment timestamps are the ones Whisper produces for each decoded segment.

    Raises InvalidAudioError for files that can't be decoded; any other
    exception is treated as temporary by the caller.
    """
    import whisper

    path = Path(path)
    if not path.exists():
        raise InvalidAudioError(f"Audio file not found: {path.name}")

    # Decode with FFmpeg first, so "bad file" is clearly separated from
    # "model failed". whisper.load_audio raises RuntimeError when FFmpeg fails.
    try:
        audio = whisper.load_audio(str(path))
    except RuntimeError as exc:
        # Don't leak the server-side path in an API-facing message.
        reason = _last_line(str(exc)).replace(str(path), path.name)
        raise InvalidAudioError(f"Could not decode audio: {reason}") from exc
    if audio.size == 0:
        raise InvalidAudioError("Audio file contains no audio data")

    model = get_model()
    result = model.transcribe(
        audio,
        language=settings.whisper_language,
        fp16=settings.whisper_device != "cpu",  # fp16 is not supported on CPU
    )

    segments = [
        {
            "start": round(float(seg["start"]), 2),
            "end": round(float(seg["end"]), 2),
            "text": seg["text"].strip(),
        }
        for seg in result.get("segments", [])
    ]
    return {
        "text": result.get("text", "").strip(),
        "language": result.get("language"),
        "segments": segments,
    }


def _last_line(message: str) -> str:
    """FFmpeg errors are long; the last non-empty line is usually the useful one."""
    lines = [line.strip() for line in message.strip().splitlines() if line.strip()]
    return lines[-1] if lines else "unknown error"
