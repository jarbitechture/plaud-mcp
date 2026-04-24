"""Audio analysis pipeline: ffmpeg → transcribe → diarize → align.

`analyze()` is a pure function. Pass injected callables for testing
without model weights or ffmpeg invocation.
"""

from __future__ import annotations

import subprocess
import tempfile
from collections.abc import Callable
from pathlib import Path

from plaud_mcp.audio.align import SpeakerTurn, align_words_to_speakers
from plaud_mcp.audio.diarize import DiarizationResult
from plaud_mcp.audio.models import AudioAnalysis, Segment
from plaud_mcp.audio.transcribe import TranscriptionResult

SAMPLE_RATE = 16_000

TranscriberFn = Callable[[Path, str | None], TranscriptionResult]
DiarizerFn = Callable[[Path, int | None], DiarizationResult]
ResamplerFn = Callable[[Path, Path], None]


def _resample_to_wav(source: Path, dest: Path) -> None:
    """Resample audio to 16kHz mono WAV using ffmpeg."""
    result = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i", str(source),
            "-ar", str(SAMPLE_RATE),
            "-ac", "1",
            "-f", "wav",
            str(dest),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed (exit {result.returncode}): {result.stderr[:500]}"
        )


def identity_resample(source: Path, dest: Path) -> None:
    """Copy source to dest without resampling. Use as resampler= in tests."""
    import shutil  # noqa: PLC0415
    shutil.copy2(source, dest)


def analyze(
    audio_path: Path,
    *,
    transcriber: TranscriberFn,
    diarizer: DiarizerFn,
    language: str | None,
    num_speakers: int | None,
    return_word_timestamps: bool,
    resampler: ResamplerFn = _resample_to_wav,  # inject identity_resample in tests
) -> AudioAnalysis:
    """Orchestrate ffmpeg → transcribe → diarize → align into AudioAnalysis.

    Raises FileNotFoundError if audio_path does not exist.
    Inject `resampler=_identity_resample` in tests to skip ffmpeg.
    """
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
        resampled = Path(tmp.name)
        resampler(audio_path, resampled)

        transcription = transcriber(resampled, language)
        diarization = diarizer(resampled, num_speakers)

    speaker_turns = [
        SpeakerTurn(speaker_id=t.speaker_id, start=t.start, end=t.end)
        for t in diarization.turns
    ]

    raw_segments = align_words_to_speakers(transcription.words, speaker_turns)

    segments: list[Segment] = [
        Segment(
            start=seg.start,
            end=seg.end,
            speaker_id=seg.speaker_id,
            text=seg.text,
            language=transcription.language,
            words=seg.words if return_word_timestamps else [],
        )
        for seg in raw_segments
    ]

    return AudioAnalysis(
        audio_path=str(audio_path),
        duration_seconds=transcription.duration_seconds,
        num_speakers=diarization.num_speakers,
        language=transcription.language,
        segments=segments,
        model_versions={
            "whisper": transcription.model_version,
            "pyannote": diarization.model_version,
        },
    )
