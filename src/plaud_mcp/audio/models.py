"""Pydantic v2 schemas for audio analysis output."""

from pydantic import BaseModel


class Word(BaseModel):
    text: str
    start: float
    end: float
    confidence: float | None = None


class Segment(BaseModel):
    start: float
    end: float
    speaker_id: str  # e.g. "SPEAKER_00"
    text: str
    language: str
    words: list[Word]


class AudioAnalysis(BaseModel):
    audio_path: str
    duration_seconds: float
    num_speakers: int
    language: str
    segments: list[Segment]
    model_versions: dict[str, str]
