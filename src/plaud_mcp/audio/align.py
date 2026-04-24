"""Align transcribed words to diarization speaker turns.

Rules:
- Each word is assigned to the turn with maximum overlap duration.
- Ties (equal overlap) go to the earlier turn (lower start time).
- Words with zero overlap to all turns go to the turn whose interval
  is nearest to the word midpoint; ties go to the earlier turn.
- Consecutive words sharing the same speaker are merged into one Segment.
"""

from dataclasses import dataclass

from plaud_mcp.audio.models import Segment, Word


@dataclass(frozen=True)
class SpeakerTurn:
    speaker_id: str
    start: float
    end: float


def _overlap(word: Word, turn: SpeakerTurn) -> float:
    return max(0.0, min(word.end, turn.end) - max(word.start, turn.start))


def _nearest_distance(word: Word, turn: SpeakerTurn) -> float:
    midpoint = (word.start + word.end) / 2.0
    # Clamp midpoint into [turn.start, turn.end] and measure distance
    clamped = max(turn.start, min(turn.end, midpoint))
    return abs(midpoint - clamped)


def _assign_word(word: Word, turns: list[SpeakerTurn]) -> str:
    best_turn = turns[0]
    best_overlap = _overlap(word, best_turn)

    for turn in turns[1:]:
        ov = _overlap(word, turn)
        if ov > best_overlap:
            best_overlap = ov
            best_turn = turn

    if best_overlap > 0.0:
        return best_turn.speaker_id

    # Zero overlap — assign by nearest midpoint; ties preserved to first (earlier) candidate
    best_turn = min(turns, key=lambda t: (_nearest_distance(word, t), t.start))
    return best_turn.speaker_id


def align_words_to_speakers(words: list[Word], speaker_turns: list[SpeakerTurn]) -> list[Segment]:
    if not words or not speaker_turns:
        return []

    assigned: list[tuple[Word, str]] = [
        (w, _assign_word(w, speaker_turns)) for w in words
    ]

    segments: list[Segment] = []
    run_words: list[Word] = [assigned[0][0]]
    run_speaker = assigned[0][1]

    def _flush(run_words: list[Word], speaker_id: str) -> Segment:
        return Segment(
            start=run_words[0].start,
            end=run_words[-1].end,
            speaker_id=speaker_id,
            text=" ".join(w.text for w in run_words),
            language="",  # filled by pipeline with detected language
            words=list(run_words),
        )

    for word, speaker_id in assigned[1:]:
        if speaker_id == run_speaker:
            run_words.append(word)
        else:
            segments.append(_flush(run_words, run_speaker))
            run_words = [word]
            run_speaker = speaker_id

    segments.append(_flush(run_words, run_speaker))
    return segments
