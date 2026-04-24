"""Tests for the analyze pipeline using injected fakes.

Verifies call ordering, output structure, and option threading
without touching any ML model.
"""

from pathlib import Path

import pytest

from plaud_mcp.audio.diarize import DiarizationResult, SpeakerInterval
from plaud_mcp.audio.models import AudioAnalysis, Word
from plaud_mcp.audio.pipeline import analyze, identity_resample
from plaud_mcp.audio.transcribe import TranscriptionResult

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


def _make_transcription(language: str = "en") -> TranscriptionResult:
    return TranscriptionResult(
        language=language,
        duration_seconds=5.0,
        words=[
            Word(text="hello", start=0.0, end=1.0, confidence=0.99),
            Word(text="world", start=1.5, end=2.5, confidence=0.95),
        ],
        model_version="large-v3",
    )


def _make_diarization() -> DiarizationResult:
    return DiarizationResult(
        num_speakers=1,
        turns=[SpeakerInterval(speaker_id="SPEAKER_00", start=0.0, end=5.0)],
        model_version="3.1.0",
    )


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_analyze_returns_audio_analysis(tmp_path: Path) -> None:
    audio = tmp_path / "test.wav"
    audio.write_bytes(b"RIFF fake")

    result = analyze(
        audio_path=audio,
        transcriber=lambda p, lang: _make_transcription(),
        diarizer=lambda p, ns: _make_diarization(),
        resampler=identity_resample,
        language=None,
        num_speakers=None,
        return_word_timestamps=True,
    )
    assert isinstance(result, AudioAnalysis)


def test_analyze_audio_path_stored(tmp_path: Path) -> None:
    audio = tmp_path / "test.wav"
    audio.write_bytes(b"RIFF fake")

    result = analyze(
        audio_path=audio,
        transcriber=lambda p, lang: _make_transcription(),
        diarizer=lambda p, ns: _make_diarization(),
        resampler=identity_resample,
        language=None,
        num_speakers=None,
        return_word_timestamps=True,
    )
    assert result.audio_path == str(audio)


def test_analyze_language_from_transcription(tmp_path: Path) -> None:
    audio = tmp_path / "test.wav"
    audio.write_bytes(b"RIFF fake")

    result = analyze(
        audio_path=audio,
        transcriber=lambda p, lang: _make_transcription(language="fr"),
        diarizer=lambda p, ns: _make_diarization(),
        resampler=identity_resample,
        language=None,
        num_speakers=None,
        return_word_timestamps=True,
    )
    assert result.language == "fr"


def test_analyze_model_versions_populated(tmp_path: Path) -> None:
    audio = tmp_path / "test.wav"
    audio.write_bytes(b"RIFF fake")

    result = analyze(
        audio_path=audio,
        transcriber=lambda p, lang: _make_transcription(),
        diarizer=lambda p, ns: _make_diarization(),
        resampler=identity_resample,
        language=None,
        num_speakers=None,
        return_word_timestamps=True,
    )
    assert "whisper" in result.model_versions
    assert "pyannote" in result.model_versions


def test_analyze_num_speakers_matches_diarization(tmp_path: Path) -> None:
    audio = tmp_path / "test.wav"
    audio.write_bytes(b"RIFF fake")

    result = analyze(
        audio_path=audio,
        transcriber=lambda p, lang: _make_transcription(),
        diarizer=lambda p, ns: _make_diarization(),
        resampler=identity_resample,
        language=None,
        num_speakers=None,
        return_word_timestamps=True,
    )
    assert result.num_speakers == 1


# ---------------------------------------------------------------------------
# Call ordering — transcriber before diarizer
# ---------------------------------------------------------------------------


def test_analyze_calls_transcriber_then_diarizer(tmp_path: Path) -> None:
    audio = tmp_path / "test.wav"
    audio.write_bytes(b"RIFF fake")

    call_order: list[str] = []

    def fake_transcribe(p: Path, lang: str | None) -> TranscriptionResult:
        call_order.append("transcribe")
        return _make_transcription()

    def fake_diarize(p: Path, ns: int | None) -> DiarizationResult:
        call_order.append("diarize")
        return _make_diarization()

    analyze(
        audio_path=audio,
        transcriber=fake_transcribe,
        diarizer=fake_diarize,
        resampler=identity_resample,
        language=None,
        num_speakers=None,
        return_word_timestamps=True,
    )
    assert call_order == ["transcribe", "diarize"]


def test_analyze_passes_language_to_transcriber(tmp_path: Path) -> None:
    audio = tmp_path / "test.wav"
    audio.write_bytes(b"RIFF fake")

    received_lang: list[str | None] = []

    def fake_transcribe(p: Path, lang: str | None) -> TranscriptionResult:
        received_lang.append(lang)
        return _make_transcription()

    analyze(
        audio_path=audio,
        transcriber=fake_transcribe,
        diarizer=lambda p, ns: _make_diarization(),
        resampler=identity_resample,
        language="ja",
        num_speakers=None,
        return_word_timestamps=True,
    )
    assert received_lang[0] == "ja"


def test_analyze_passes_num_speakers_to_diarizer(tmp_path: Path) -> None:
    audio = tmp_path / "test.wav"
    audio.write_bytes(b"RIFF fake")

    received_ns: list[int | None] = []

    def fake_diarize(p: Path, ns: int | None) -> DiarizationResult:
        received_ns.append(ns)
        return _make_diarization()

    analyze(
        audio_path=audio,
        transcriber=lambda p, lang: _make_transcription(),
        diarizer=fake_diarize,
        resampler=identity_resample,
        language=None,
        num_speakers=3,
        return_word_timestamps=True,
    )
    assert received_ns[0] == 3


# ---------------------------------------------------------------------------
# Word timestamps option
# ---------------------------------------------------------------------------


def test_analyze_word_timestamps_false_strips_words(tmp_path: Path) -> None:
    audio = tmp_path / "test.wav"
    audio.write_bytes(b"RIFF fake")

    result = analyze(
        audio_path=audio,
        transcriber=lambda p, lang: _make_transcription(),
        diarizer=lambda p, ns: _make_diarization(),
        resampler=identity_resample,
        language=None,
        num_speakers=None,
        return_word_timestamps=False,
    )
    for seg in result.segments:
        assert seg.words == []


def test_analyze_word_timestamps_true_keeps_words(tmp_path: Path) -> None:
    audio = tmp_path / "test.wav"
    audio.write_bytes(b"RIFF fake")

    result = analyze(
        audio_path=audio,
        transcriber=lambda p, lang: _make_transcription(),
        diarizer=lambda p, ns: _make_diarization(),
        resampler=identity_resample,
        language=None,
        num_speakers=None,
        return_word_timestamps=True,
    )
    assert any(len(seg.words) > 0 for seg in result.segments)


# ---------------------------------------------------------------------------
# Missing file
# ---------------------------------------------------------------------------


def test_analyze_missing_file_raises(tmp_path: Path) -> None:
    missing = tmp_path / "nonexistent.wav"
    with pytest.raises(FileNotFoundError):
        analyze(
            audio_path=missing,
            transcriber=lambda p, lang: _make_transcription(),
            diarizer=lambda p, ns: _make_diarization(),
        resampler=identity_resample,
            language=None,
            num_speakers=None,
            return_word_timestamps=True,
        )
