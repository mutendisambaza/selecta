from selecta.features.types import PhraseRegion, TrackFeatures
from selecta.scoring.scorer import TransitionScore, score


def _track(
    path: str,
    bpm: float,
    key_camelot: str,
    intensity: float,
    spectral: dict[str, dict[str, float]],
    phrases: tuple[PhraseRegion, ...] = (),
) -> TrackFeatures:
    return TrackFeatures(
        path=path,
        duration=240.0,
        bpm=bpm,
        bpm_confidence=1.0,
        key_camelot=key_camelot,
        key_confidence=1.0,
        intensity=intensity,
        spectral=spectral,
        phrases=phrases,
    )


def _phrase(label: str) -> PhraseRegion:
    return PhraseRegion(start=0.0, end=16.0, bars=8, label=label, confidence=1.0)


def test_score_returns_transition_score_with_high_quality_match():
    weights = {key: 0.2 for key in ("harmonic", "tempo", "energy", "spectral", "phrase")}
    a = _track(
        path="/music/a.wav",
        bpm=128.0,
        key_camelot="8A",
        intensity=0.65,
        spectral={"outro": {"low": 1.0, "low_mid": 0.0, "high_mid": 0.0, "high": 0.0}},
        phrases=(_phrase("outro"),),
    )
    b = _track(
        path="/music/b.wav",
        bpm=128.0,
        key_camelot="8A",
        intensity=0.75,
        spectral={"intro": {"low": 0.0, "low_mid": 0.0, "high_mid": 0.0, "high": 1.0}},
        phrases=(_phrase("intro"),),
    )

    result = score(a, b, arc_target=0.75, weights=weights)

    assert isinstance(result, TransitionScore)
    assert result.total > 80.0
    assert set(result.breakdown) == {"harmonic", "tempo", "energy", "spectral", "phrase"}
    assert all(0.0 <= value <= 1.0 for value in result.breakdown.values())


def test_score_returns_low_total_for_poor_transition_match():
    weights = {key: 0.2 for key in ("harmonic", "tempo", "energy", "spectral", "phrase")}
    a = _track(
        path="/music/a.wav",
        bpm=128.0,
        key_camelot="8A",
        intensity=0.8,
        spectral={"body": {"low": 1.0, "low_mid": 0.0, "high_mid": 0.0, "high": 0.0}},
    )
    b = _track(
        path="/music/b.wav",
        bpm=150.0,
        key_camelot="2A",
        intensity=0.95,
        spectral={"body": {"low": 1.0, "low_mid": 0.0, "high_mid": 0.0, "high": 0.0}},
    )

    result = score(a, b, arc_target=0.1, weights=weights)

    assert isinstance(result, TransitionScore)
    assert result.total < 40.0
    assert set(result.breakdown) == {"harmonic", "tempo", "energy", "spectral", "phrase"}
    assert all(0.0 <= value <= 1.0 for value in result.breakdown.values())
