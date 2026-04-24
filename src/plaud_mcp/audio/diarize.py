"""Speaker diarization via pyannote.audio.

Requires HF_TOKEN env var for gated model weights.
Lazy-loads the pipeline on first call.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PYANNOTE_MODEL = "pyannote/speaker-diarization-3.1"
_PYANNOTE_VERSION = "3.1.0"

_pipeline_cache: dict[str, Any] = {}


@dataclass(frozen=True)
class SpeakerInterval:
    speaker_id: str
    start: float
    end: float


@dataclass
class DiarizationResult:
    num_speakers: int
    turns: list[SpeakerInterval]
    model_version: str = _PYANNOTE_VERSION


def _get_pipeline(model_id: str) -> Any:
    if model_id not in _pipeline_cache:
        token = os.environ.get("HF_TOKEN")
        if not token:
            raise RuntimeError(
                "HF_TOKEN environment variable is required for pyannote.audio diarization. "
                "Set it to your Hugging Face access token with pyannote/speaker-diarization-3.1 access."
            )
        from pyannote.audio import Pipeline  # type: ignore[import-untyped]  # noqa: PLC0415

        _pipeline_cache[model_id] = Pipeline.from_pretrained(model_id, use_auth_token=token)  # type: ignore[no-untyped-call]
    return _pipeline_cache[model_id]


def diarize(audio_path: Path, num_speakers: int | None = None) -> DiarizationResult:
    """Run speaker diarization on audio."""
    pipeline = _get_pipeline(PYANNOTE_MODEL)

    kwargs: dict[str, Any] = {}
    if num_speakers is not None:
        kwargs["num_speakers"] = num_speakers

    annotation = pipeline(str(audio_path), **kwargs)

    turns: list[SpeakerInterval] = []
    for segment, _, speaker in annotation.itertracks(yield_label=True):
        turns.append(
            SpeakerInterval(
                speaker_id=speaker,
                start=float(segment.start),
                end=float(segment.end),
            )
        )

    detected_speakers = len({t.speaker_id for t in turns})

    return DiarizationResult(
        num_speakers=detected_speakers,
        turns=turns,
        model_version=_PYANNOTE_VERSION,
    )
