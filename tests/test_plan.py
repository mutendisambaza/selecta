import json

import pytest

from selecta.features.types import PhraseRegion, TrackFeatures
from selecta.render.plan import render
from selecta.sequencing.sequencer import OrderedTrack


def _phrase(
    label: str,
    start: float,
    end: float,
    bars: int,
    confidence: float = 1.0,
) -> PhraseRegion:
    return PhraseRegion(
        start=start,
        end=end,
        bars=bars,
        label=label,
        confidence=confidence,
    )


def _track(
    path: str,
    title: str,
    artist: str,
    bpm: float,
    key_camelot: str,
    intensity: float,
    duration: float,
    phrases: tuple[PhraseRegion, ...],
) -> TrackFeatures:
    return TrackFeatures(
        path=path,
        duration=duration,
        bpm=bpm,
        bpm_confidence=1.0,
        key_camelot=key_camelot,
        key_confidence=1.0,
        intensity=intensity,
        phrases=phrases,
        title=title,
        artist=artist,
    )


def _ordered() -> list[OrderedTrack]:
    first = _track(
        path="/music/opening.wav",
        title="Opening Pressure",
        artist="Aster",
        bpm=120.0,
        key_camelot="8A",
        intensity=0.32,
        duration=240.0,
        phrases=(
            _phrase("body", 16.0, 48.0, 16),
            _phrase("outro", 192.0, 224.0, 16),
        ),
    )
    second = _track(
        path="/music/lift.wav",
        title="Lift Signal",
        artist="Bex",
        bpm=124.0,
        key_camelot="9A",
        intensity=0.54,
        duration=260.0,
        phrases=(
            _phrase("intro", 32.0, 64.0, 16, confidence=0.9),
            _phrase("body", 64.0, 128.0, 32),
        ),
    )
    third = _track(
        path="/music/peak.wav",
        title="Peak Thread",
        artist="Ciro",
        bpm=126.0,
        key_camelot="9B",
        intensity=0.78,
        duration=280.0,
        phrases=(
            _phrase("intro", 48.0, 72.0, 12, confidence=0.4),
            _phrase("drop", 72.0, 104.0, 16),
        ),
    )
    return [
        OrderedTrack(
            track=first,
            position=0,
            arc_pos=0.0,
            arc_target=0.30,
            incoming_score=None,
        ),
        OrderedTrack(
            track=second,
            position=1,
            arc_pos=0.5,
            arc_target=0.55,
            incoming_score=78.4,
        ),
        OrderedTrack(
            track=third,
            position=2,
            arc_pos=1.0,
            arc_target=0.80,
            incoming_score=22.0,
        ),
    ]


def test_render_markdown_includes_tracks_and_transitions():
    output = render(_ordered(), "markdown")

    assert output.startswith("# Set Plan")
    assert "Opening Pressure" in output
    assert "Lift Signal" in output
    assert "Peak Thread" in output
    assert output.count("↳") == 2
    assert "BPM +3.3%" in output
    assert "BPM +1.6%" in output
    assert "bring in at 1:04" in output  # intro END (groove entry), not intro start
    assert "swap bass at 1:04" in output
    assert "low confidence — trust your ears" in output


def test_render_json_returns_loadable_transition_data():
    payload = json.loads(render(_ordered(), "json"))

    assert len(payload) == 3
    assert payload[0]["transition"] is None

    second_transition = payload[1]["transition"]
    assert second_transition == {
        "from_key": "8A",
        "to_key": "9A",
        "bpm_pct": pytest.approx(3.3333333333333335),
        "intro_cue": "1:04",
        "outro_cue": "3:12",
        "swap_bass_at": "1:04",
    }

    third_transition = payload[2]["transition"]
    assert third_transition["from_key"] == "9A"
    assert third_transition["to_key"] == "9B"
    assert third_transition["intro_cue"] == "1:12"  # intro END (groove entry)
    assert third_transition["outro_cue"] == "4:20"
    assert third_transition["swap_bass_at"] == "1:20"
    assert third_transition["confidence_note"] == "low confidence — trust your ears"


def test_render_rejects_unknown_format():
    with pytest.raises(ValueError):
        render(_ordered(), "xml")


def test_bring_in_at_uses_intro_end():
    from selecta.features.types import PhraseRegion, TrackFeatures
    from selecta.sequencing.sequencer import OrderedTrack
    from selecta.render.plan import render
    import json

    def mk(path, intro_end):
        return TrackFeatures(path=path, duration=300, bpm=120, bpm_confidence=1,
            beats=(), key_camelot="8A", key_confidence=1, energy_curve=(), intensity=0.5,
            spectral={}, phrases=(PhraseRegion(0.0, intro_end, 16, "intro", 0.9),
                                   PhraseRegion(intro_end, 300, 32, "body", 0.9)))
    a = OrderedTrack(track=mk("a", 16.0), position=0, arc_pos=0.0, arc_target=0.2, incoming_score=None)
    b = OrderedTrack(track=mk("b", 16.0), position=1, arc_pos=1.0, arc_target=0.5, incoming_score=80.0)
    data = json.loads(render([a, b], "json"))
    assert data[1]["transition"]["intro_cue"] == "0:16"   # intro END, not 0:00
