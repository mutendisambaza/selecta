"""Render a sequenced set into human and machine-readable plans."""

from __future__ import annotations

import json

from selecta.features.types import PhraseRegion
from selecta.sequencing.sequencer import OrderedTrack


def render(ordered: list[OrderedTrack], fmt: str = "markdown") -> str:
    """Render an ordered set plan in markdown or JSON."""

    items = [
        _plan_item(item, ordered[index - 1] if index > 0 else None)
        for index, item in enumerate(ordered)
    ]

    if fmt == "json":
        return json.dumps(items)
    if fmt == "markdown":
        return _render_markdown(items)
    raise ValueError(f"unknown format: {fmt}")


def _plan_item(item: OrderedTrack, previous: OrderedTrack | None) -> dict[str, object]:
    track = item.track
    return {
        "position": item.position,
        "title": track.title,
        "artist": track.artist,
        "bpm": track.bpm,
        "key_camelot": track.key_camelot,
        "intensity": track.intensity,
        "arc_target": item.arc_target,
        "incoming_score": item.incoming_score,
        "transition": None if previous is None else _transition(previous, item),
    }


def _transition(previous: OrderedTrack, current: OrderedTrack) -> dict[str, object]:
    prev_track = previous.track
    track = current.track
    intro_phrase = _first_phrase(track.phrases, "intro")
    outro_phrase = _first_phrase(prev_track.phrases, "outro")
    intro_start = intro_phrase.start if intro_phrase is not None else 0.0
    outro_start = outro_phrase.start if outro_phrase is not None else prev_track.duration
    bpm_pct = 0.0
    if prev_track.bpm > 0:
        bpm_pct = ((track.bpm - prev_track.bpm) / prev_track.bpm) * 100.0

    transition = {
        "from_key": prev_track.key_camelot,
        "to_key": track.key_camelot,
        "bpm_pct": bpm_pct,
        "intro_cue": _format_timestamp(intro_start),
        "outro_cue": _format_timestamp(outro_start),
    }

    swap_bass_at = _swap_bass_at(intro_phrase)
    if swap_bass_at is not None:
        transition["swap_bass_at"] = swap_bass_at

    low_confidence = (
        (intro_phrase is not None and intro_phrase.confidence < 0.5)
        or (current.incoming_score is not None and current.incoming_score < 25.0)
    )
    if low_confidence:
        transition["confidence_note"] = "low confidence — trust your ears"

    return transition


def _render_markdown(items: list[dict[str, object]]) -> str:
    lines = ["# Set Plan", ""]

    for item in items:
        lines.append(
            (
                f"{int(item['position']) + 1}. {item['title']} — {item['artist']} | "
                f"{item['key_camelot']} | {float(item['bpm']):.1f} BPM | "
                f"energy {float(item['intensity']):.2f} (target {float(item['arc_target']):.2f})"
            )
        )
        transition = item["transition"]
        if transition is not None:
            line = (
                f"   ↳ from {transition['from_key']} to {transition['to_key']} | "
                f"BPM {float(transition['bpm_pct']):+,.1f}% | "
                f"bring in at {transition['intro_cue']}"
            )
            if "swap_bass_at" in transition:
                line += f" | swap bass at {transition['swap_bass_at']}"
            line += f" | mix score {float(item['incoming_score']):.0f}"
            if "confidence_note" in transition:
                line += f" ({transition['confidence_note']})"
            lines.append(line)

    return "\n".join(lines)


def _first_phrase(phrases: tuple[PhraseRegion, ...], label: str) -> PhraseRegion | None:
    for phrase in phrases:
        if phrase.label == label:
            return phrase
    return None


def _swap_bass_at(intro_phrase: PhraseRegion | None) -> str | None:
    if intro_phrase is None:
        return None
    phrase_duration = intro_phrase.end - intro_phrase.start
    if intro_phrase.bars <= 0 or phrase_duration <= 0:
        return None
    seconds_per_bar = phrase_duration / intro_phrase.bars
    return _format_timestamp(intro_phrase.start + (seconds_per_bar * 16.0))


def _format_timestamp(seconds: float) -> str:
    total_seconds = max(0, int(seconds))
    minutes, remainder = divmod(total_seconds, 60)
    return f"{minutes}:{remainder:02d}"
