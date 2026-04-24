"""ASR via mlx-whisper.

Lazy-loads the model on first call. Monkeypatch `transcribe` in tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from plaud_mcp.audio.models import Word

SAMPLE_RATE = 16_000
_DEFAULT_MODEL = "mlx-community/whisper-large-v3-mlx"
_WHISPER_VERSION = "large-v3"

_model_cache: dict[str, Any] = {}


@dataclass
class TranscriptionResult:
    language: str
    duration_seconds: float
    words: list[Word]
    model_version: str = _WHISPER_VERSION


def _get_model(model_id: str) -> Any:
    if model_id not in _model_cache:
        import mlx_whisper  # type: ignore[import-untyped]  # noqa: PLC0415

        _model_cache[model_id] = mlx_whisper
    return _model_cache[model_id]


def transcribe(audio_path: Path, language: str | None = None) -> TranscriptionResult:
    """Transcribe audio using mlx-whisper with word-level timestamps."""
    mlx_whisper = _get_model(_DEFAULT_MODEL)

    raw: dict[str, Any] = mlx_whisper.transcribe(
        str(audio_path),
        path_or_hf_repo=_DEFAULT_MODEL,
        language=language,
        word_timestamps=True,
    )

    detected_language: str = raw.get("language", language or "en")
    segments_raw: list[dict[str, Any]] = raw.get("segments", [])

    words: list[Word] = []
    for seg in segments_raw:
        for w in seg.get("words", []):
            words.append(
                Word(
                    text=w["word"].strip(),
                    start=float(w["start"]),
                    end=float(w["end"]),
                    confidence=float(w["probability"]) if "probability" in w else None,
                )
            )

    duration = float(segments_raw[-1]["end"]) if segments_raw else 0.0

    return TranscriptionResult(
        language=detected_language,
        duration_seconds=duration,
        words=words,
        model_version=_WHISPER_VERSION,
    )
