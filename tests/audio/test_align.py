"""Tests for word-to-speaker alignment.

Alignment rules (encoded here; implementation must match):
- Each word is assigned to the speaker turn with maximum overlap duration.
- Ties go to the earlier turn (lower start time).
- A word with zero overlap to every turn is assigned by nearest midpoint; ties go to earlier turn.
"""

import pytest

from plaud_mcp.audio.align import SpeakerTurn, align_words_to_speakers
from plaud_mcp.audio.models import Word

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def word(text: str, start: float, end: float, confidence: float | None = None) -> Word:
    return Word(text=text, start=start, end=end, confidence=confidence)


def turn(speaker_id: str, start: float, end: float) -> SpeakerTurn:
    return SpeakerTurn(speaker_id=speaker_id, start=start, end=end)


# ---------------------------------------------------------------------------
# Single-speaker
# ---------------------------------------------------------------------------


def test_single_speaker_all_words_assigned() -> None:
    words = [word("hello", 0.0, 1.0), word("world", 1.0, 2.0)]
    turns = [turn("SPEAKER_00", 0.0, 2.0)]
    segments = align_words_to_speakers(words, turns)
    assert len(segments) == 1
    assert segments[0].speaker_id == "SPEAKER_00"
    assert len(segments[0].words) == 2
    assert segments[0].text == "hello world"


def test_single_speaker_text_joined_with_spaces() -> None:
    words = [word("a", 0.0, 0.5), word("b", 0.5, 1.0), word("c", 1.0, 1.5)]
    turns = [turn("SPEAKER_00", 0.0, 2.0)]
    segments = align_words_to_speakers(words, turns)
    assert segments[0].text == "a b c"


# ---------------------------------------------------------------------------
# Two speakers — clean boundary
# ---------------------------------------------------------------------------


def test_two_speakers_clean_split() -> None:
    words = [
        word("first", 0.0, 1.0),
        word("second", 1.0, 2.0),
        word("third", 2.0, 3.0),
        word("fourth", 3.0, 4.0),
    ]
    turns = [turn("SPEAKER_00", 0.0, 2.0), turn("SPEAKER_01", 2.0, 4.0)]
    segments = align_words_to_speakers(words, turns)
    assert len(segments) == 2
    assert segments[0].speaker_id == "SPEAKER_00"
    assert segments[0].text == "first second"
    assert segments[1].speaker_id == "SPEAKER_01"
    assert segments[1].text == "third fourth"


def test_segment_timestamps_match_word_boundaries() -> None:
    words = [word("hello", 0.5, 1.0), word("world", 1.5, 2.0)]
    turns = [turn("SPEAKER_00", 0.0, 3.0)]
    segments = align_words_to_speakers(words, turns)
    assert segments[0].start == 0.5
    assert segments[0].end == 2.0


# ---------------------------------------------------------------------------
# Overlap — word straddles a speaker boundary
# ---------------------------------------------------------------------------


def test_word_overlap_majority_wins() -> None:
    # word spans 1.0–2.0; SPEAKER_00 covers 0.0–1.3 (0.3s overlap),
    # SPEAKER_01 covers 1.3–3.0 (0.7s overlap) → SPEAKER_01 wins
    words = [word("bridge", 1.0, 2.0)]
    turns = [turn("SPEAKER_00", 0.0, 1.3), turn("SPEAKER_01", 1.3, 3.0)]
    segments = align_words_to_speakers(words, turns)
    assert len(segments) == 1
    assert segments[0].speaker_id == "SPEAKER_01"


def test_word_overlap_minority_loses() -> None:
    # word spans 0.0–1.0; SPEAKER_00 covers 0.0–0.9 (0.9s overlap),
    # SPEAKER_01 covers 0.9–2.0 (0.1s overlap) → SPEAKER_00 wins
    words = [word("early", 0.0, 1.0)]
    turns = [turn("SPEAKER_00", 0.0, 0.9), turn("SPEAKER_01", 0.9, 2.0)]
    segments = align_words_to_speakers(words, turns)
    assert segments[0].speaker_id == "SPEAKER_00"


def test_tied_overlap_earlier_turn_wins() -> None:
    # word spans 0.0–2.0; both turns overlap by exactly 1.0s → earlier (SPEAKER_00) wins
    words = [word("tie", 0.0, 2.0)]
    turns = [turn("SPEAKER_00", 0.0, 1.0), turn("SPEAKER_01", 1.0, 2.0)]
    segments = align_words_to_speakers(words, turns)
    assert segments[0].speaker_id == "SPEAKER_00"


# ---------------------------------------------------------------------------
# Gaps — word falls outside all turns
# ---------------------------------------------------------------------------


def test_gap_word_assigned_to_nearest_by_midpoint() -> None:
    # word midpoint is 2.5; SPEAKER_00 ends at 1.0, SPEAKER_01 starts at 4.0
    # distance: |2.5 - 1.0| = 1.5 vs |2.5 - 4.0| = 1.5 → tie → earlier turn wins
    words = [word("silence", 2.0, 3.0)]
    turns = [turn("SPEAKER_00", 0.0, 1.0), turn("SPEAKER_01", 4.0, 5.0)]
    segments = align_words_to_speakers(words, turns)
    assert segments[0].speaker_id == "SPEAKER_00"


def test_gap_word_closer_to_later_turn() -> None:
    # word midpoint 3.8; SPEAKER_00 ends at 1.0 (dist 2.8), SPEAKER_01 starts at 3.0 (dist 0.8)
    words = [word("close", 3.6, 4.0)]
    turns = [turn("SPEAKER_00", 0.0, 1.0), turn("SPEAKER_01", 3.0, 3.5)]
    segments = align_words_to_speakers(words, turns)
    assert segments[0].speaker_id == "SPEAKER_01"


def test_no_turns_returns_empty() -> None:
    words = [word("orphan", 0.0, 1.0)]
    segments = align_words_to_speakers(words, [])
    assert segments == []


def test_no_words_returns_empty() -> None:
    turns = [turn("SPEAKER_00", 0.0, 5.0)]
    segments = align_words_to_speakers([], turns)
    assert segments == []


# ---------------------------------------------------------------------------
# Consecutive same-speaker words merge into one segment
# ---------------------------------------------------------------------------


def test_same_speaker_runs_merge() -> None:
    # Two words for S00, then two for S01, then one for S00 → 3 segments
    words = [
        word("a", 0.0, 1.0),
        word("b", 1.0, 2.0),
        word("c", 2.0, 3.0),
        word("d", 3.0, 4.0),
        word("e", 4.0, 5.0),
    ]
    turns = [turn("SPEAKER_00", 0.0, 2.0), turn("SPEAKER_01", 2.0, 4.0), turn("SPEAKER_00", 4.0, 5.0)]
    segments = align_words_to_speakers(words, turns)
    assert len(segments) == 3
    assert segments[0].speaker_id == "SPEAKER_00"
    assert segments[1].speaker_id == "SPEAKER_01"
    assert segments[2].speaker_id == "SPEAKER_00"


def test_segment_word_list_preserved() -> None:
    words = [word("x", 0.0, 0.5, confidence=0.9), word("y", 0.5, 1.0, confidence=0.8)]
    turns = [turn("SPEAKER_00", 0.0, 1.0)]
    segments = align_words_to_speakers(words, turns)
    assert segments[0].words[0].confidence == pytest.approx(0.9)
    assert segments[0].words[1].confidence == pytest.approx(0.8)


def test_single_word_single_turn() -> None:
    segments = align_words_to_speakers([word("solo", 1.0, 2.0)], [turn("SPEAKER_00", 0.0, 5.0)])
    assert len(segments) == 1
    assert segments[0].text == "solo"
    assert segments[0].start == 1.0
    assert segments[0].end == 2.0
