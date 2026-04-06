#!/usr/bin/env python3
"""Build organized transcript markdown files from Plaud API data.

Run from project root:
    .venv/bin/python transcripts/_build.py
"""

import asyncio
import re
import sys
from datetime import UTC, datetime
from pathlib import Path

# Add project src to path so we can import plaud_client
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from plaud_mcp.plaud_client import PlaudClient, PlaudAPIError

TRANSCRIPTS_DIR = Path(__file__).resolve().parent
DAYS = 14

# Keyword-to-tag mapping: if keyword appears in title or first 500 chars, add the tag
TAG_KEYWORDS: dict[str, list[str]] = {
    "team-sync": ["team", "sync", "standup", "stand-up", "huddle"],
    "AI": ["ai", "artificial intelligence", "machine learning", "llm", "chatbot", "gpt", "claude"],
    "MDM": ["mdm", "mobile device", "device management", "intune", "jamf"],
    "policy": ["policy", "policies", "compliance", "regulation", "governance"],
    "scrum": ["scrum", "sprint", "retrospective", "retro", "backlog", "grooming"],
    "prompt-engineering": ["prompt", "prompting", "prompt engineering"],
    "office-move": ["office move", "moving offices", "relocation", "new office"],
    "budget": ["budget", "funding", "cost", "expense", "fiscal"],
    "Power-BI": ["power bi", "powerbi", "dashboard", "report"],
    "1-on-1": ["one on one", "1-on-1", "1:1", "one-on-one", "check-in"],
    "meeting": ["meeting", "agenda", "minutes"],
    "interview": ["interview", "candidate", "hiring"],
    "training": ["training", "workshop", "learning", "onboarding"],
    "security": ["security", "cybersecurity", "vulnerability", "phishing", "mfa"],
    "infrastructure": ["infrastructure", "server", "network", "cloud", "azure", "aws"],
    "project-update": ["project update", "status update", "progress"],
    "presentation": ["presentation", "demo", "showcase"],
}

# Keywords that suggest personal content
NOTEPIN_SERIAL_PREFIX = "8820B"  # NotePin hardware serial starts with this


def detect_device(file: dict) -> str:
    """Determine device from scene code and serial number."""
    scene = file.get("scene")
    serial = file.get("serial_number", "")
    if scene == 1 or serial.startswith(NOTEPIN_SERIAL_PREFIX):
        return "NotePin"
    return "Desktop"


PERSONAL_KEYWORDS = [
    "apartment", "moving", "lease", "landlord", "rent",
    "relationship", "dating", "wedding", "divorce",
    "finances", "mortgage", "loan", "credit",
    "doctor", "dentist", "medical", "appointment",
    "vacation", "holiday", "travel plan",
    "family", "kids", "daycare",
    "gym", "workout", "diet",
]


def sanitize_filename(title: str) -> str:
    """Turn a title into a filesystem-safe string."""
    s = title.lower().strip()
    s = re.sub(r"[^\w\s-]", "", s)
    s = re.sub(r"[\s_]+", "_", s)
    s = s.strip("_")
    return s[:80] if s else "untitled"


def format_duration(ms: int | None) -> str:
    if not ms:
        return "0s"
    seconds = ms // 1000
    minutes, secs = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes}m {secs}s"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def generate_tags(title: str, transcript_preview: str) -> list[str]:
    """Scan title and first 500 chars of transcript for keyword matches."""
    search_text = (title + " " + transcript_preview[:500]).lower()
    tags = []
    for tag, keywords in TAG_KEYWORDS.items():
        if any(kw in search_text for kw in keywords):
            tags.append(tag)
    return sorted(set(tags))


def classify_type(title: str, transcript_text: str, duration_ms: int | None) -> str:
    """Classify as work, personal, or trivial."""
    # Trivial: under 30 seconds or minimal content
    if duration_ms is not None and duration_ms < 30_000:
        return "trivial"
    if len(transcript_text.strip()) < 50:
        return "trivial"

    # Personal: check for personal keywords
    search_text = (title + " " + transcript_text[:1000]).lower()
    personal_hits = sum(1 for kw in PERSONAL_KEYWORDS if kw in search_text)
    if personal_hits >= 2:
        return "personal"

    return "work"


def extract_speakers(segments: list[dict]) -> list[str]:
    """Extract unique speaker names from transcript segments."""
    speakers = []
    seen = set()
    for seg in segments:
        speaker = seg.get("speaker", "").strip()
        if speaker and speaker not in seen:
            seen.add(speaker)
            speakers.append(speaker)
    return speakers


def build_transcript_text(segments: list[dict]) -> str:
    """Build readable transcript from segments."""
    lines = []
    for seg in segments:
        speaker = seg.get("speaker", "").strip()
        content = seg.get("content", "").strip()
        if not content:
            continue
        if speaker:
            lines.append(f"**{speaker}:** {content}")
        else:
            lines.append(content)
    return "\n\n".join(lines)


def write_markdown(
    file_id: str,
    title: str,
    date_str: str,
    duration: str,
    device: str,
    speakers: list[str],
    tags: list[str],
    file_type: str,
    summary_text: str,
    transcript_text: str,
    output_path: Path,
) -> None:
    speakers_yaml = ", ".join(speakers) if speakers else ""
    tags_yaml = ", ".join(tags) if tags else ""

    # Escape quotes in title for YAML
    safe_title = title.replace('"', '\\"')

    content = f"""---
id: {file_id}
title: "{safe_title}"
date: {date_str}
duration: "{duration}"
device: {device}
speakers: [{speakers_yaml}]
tags: [{tags_yaml}]
type: {file_type}
---

## Summary

{summary_text}

## Transcript

{transcript_text}
"""
    output_path.write_text(content, encoding="utf-8")


async def process_file(client: PlaudClient, file: dict, results: list[dict]) -> None:
    """Process a single file: fetch transcript and summary, write markdown."""
    file_id = file.get("id", "")
    title = file.get("filename", "Untitled")
    start_time = file.get("start_time")
    duration_ms = file.get("duration")

    if not file_id:
        print(f"  SKIP: no file ID for '{title}'")
        return

    # Parse date
    if start_time:
        dt = datetime.fromtimestamp(start_time / 1000, tz=UTC)
        date_str = dt.strftime("%Y-%m-%d")
    else:
        date_str = "unknown"

    duration = format_duration(duration_ms)
    print(f"  Processing: {title} ({date_str}, {duration})")

    # Fetch transcript
    segments: list[dict] = []
    transcript_text = ""
    try:
        data = await client.get_transcript(file_id)
        if isinstance(data, list):
            segments = data
            transcript_text = build_transcript_text(segments)
        else:
            transcript_text = str(data)
    except Exception as e:
        transcript_text = f"*Transcript unavailable: {e}*"
        print(f"    No transcript: {e}")

    # Fetch summary
    summary_text = "No summary available"
    try:
        summary_data = await client.get_summary(file_id)
        ai_content = summary_data.get("ai_content", "")
        if ai_content:
            summary_text = ai_content
    except Exception:
        pass

    # Extract metadata
    speakers = extract_speakers(segments)
    tags = generate_tags(title, transcript_text)
    file_type = classify_type(title, transcript_text, duration_ms)
    device = detect_device(file)

    # Write markdown
    safe_name = sanitize_filename(title)
    filename = f"{date_str}_{safe_name}.md"
    output_path = TRANSCRIPTS_DIR / filename

    write_markdown(
        file_id=file_id,
        title=title,
        date_str=date_str,
        duration=duration,
        device=device,
        speakers=speakers,
        tags=tags,
        file_type=file_type,
        summary_text=summary_text,
        transcript_text=transcript_text,
        output_path=output_path,
    )

    results.append({
        "file_id": file_id,
        "title": title,
        "date": date_str,
        "duration": duration,
        "device": device,
        "speakers": speakers,
        "tags": tags,
        "type": file_type,
        "filename": filename,
    })
    print(f"    Wrote: {filename}")


def write_index(results: list[dict]) -> None:
    """Write INDEX.md with a table of all files sorted by date descending."""
    results.sort(key=lambda r: r["date"], reverse=True)

    lines = [
        "# Plaud Transcripts Index",
        "",
        f"Generated: {datetime.now(UTC).strftime('%Y-%m-%d %H:%M UTC')}",
        f"Total files: {len(results)}",
        "",
        "| Date | Title | Duration | Device | Type | Tags | Speakers |",
        "|------|-------|----------|--------|------|------|----------|",
    ]

    for r in results:
        link = f"[{r['title']}]({r['filename']})"
        tags = ", ".join(r["tags"]) if r["tags"] else "-"
        speakers = ", ".join(r["speakers"]) if r["speakers"] else "-"
        device = r.get("device", "?")
        lines.append(
            f"| {r['date']} | {link} | {r['duration']} | {device} | {r['type']} | {tags} | {speakers} |"
        )

    lines.append("")
    index_path = TRANSCRIPTS_DIR / "INDEX.md"
    index_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWrote INDEX.md with {len(results)} entries")


async def main() -> None:
    print(f"Fetching Plaud files from the last {DAYS} days...")
    client = PlaudClient()

    if not client.is_available():
        print("ERROR: Plaud Desktop not available. Is it installed and signed in?")
        sys.exit(1)

    files = await client.get_recent_files(days=DAYS)
    print(f"Found {len(files)} files\n")

    if not files:
        print("No files found in the specified period.")
        return

    results: list[dict] = []

    # Process files with concurrency limit to avoid hammering the API
    sem = asyncio.Semaphore(3)

    async def bounded_process(file: dict) -> None:
        async with sem:
            await process_file(client, file, results)

    await asyncio.gather(*[bounded_process(f) for f in files])

    write_index(results)
    print("\nDone.")


if __name__ == "__main__":
    asyncio.run(main())
