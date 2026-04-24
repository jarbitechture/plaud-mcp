"""Tests for Pydantic v2 audio schemas."""

import pytest
from pydantic import ValidationError

from plaud_mcp.audio.models import AudioAnalysis, Segment, Word


def test_word_required_fields() -> None:
    w = Word(text="hello", start=0.0, end=1.0)
    assert w.text == "hello"
    assert w.confidence is None


def test_word_with_confidence() -> None:
    w = Word(text="hi", start=0.0, end=0.5, confidence=0.95)
    assert w.confidence == pytest.approx(0.95)


def test_word_missing_required_raises() -> None:
    with pytest.raises(ValidationError):
        Word(text="oops")  # type: ignore[call-arg]


def test_word_end_before_start_is_valid() -> None:
    # Models don't enforce temporal ordering — that's pipeline responsibility
    w = Word(text="x", start=2.0, end=1.0)
    assert w.start == 2.0


def test_segment_required_fields() -> None:
    s = Segment(
        start=0.0,
        end=5.0,
        speaker_id="SPEAKER_00",
        text="hello world",
        language="en",
        words=[],
    )
    assert s.speaker_id == "SPEAKER_00"
    assert s.words == []


def test_segment_missing_speaker_raises() -> None:
    with pytest.raises(ValidationError):
        Segment(start=0.0, end=1.0, text="hi", language="en", words=[])  # type: ignore[call-arg]


def test_audio_analysis_round_trip() -> None:
    analysis = AudioAnalysis(
        audio_path="/tmp/test.wav",
        duration_seconds=30.0,
        num_speakers=2,
        language="en",
        segments=[],
        model_versions={"whisper": "large-v3", "pyannote": "3.1"},
    )
    dumped = analysis.model_dump()
    restored = AudioAnalysis.model_validate(dumped)
    assert restored.audio_path == analysis.audio_path
    assert restored.num_speakers == 2


def test_audio_analysis_json_schema_valid() -> None:
    schema = AudioAnalysis.model_json_schema()
    assert schema["type"] == "object"
    assert "segments" in schema["properties"]
    assert "audio_path" in schema["properties"]


def test_audio_analysis_segments_nested() -> None:
    w = Word(text="hi", start=0.0, end=0.5)
    s = Segment(start=0.0, end=0.5, speaker_id="SPEAKER_00", text="hi", language="en", words=[w])
    analysis = AudioAnalysis(
        audio_path="/tmp/a.wav",
        duration_seconds=1.0,
        num_speakers=1,
        language="en",
        segments=[s],
        model_versions={},
    )
    dumped = analysis.model_dump()
    assert dumped["segments"][0]["words"][0]["text"] == "hi"


def test_audio_analysis_missing_required_raises() -> None:
    with pytest.raises(ValidationError):
        AudioAnalysis(audio_path="/tmp/a.wav")  # type: ignore[call-arg]
