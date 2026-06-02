from statistics import median

from selecta.features.types import PhraseRegion, TrackFeatures
from selecta.sequencing.arc import arc_target, load_profile
from selecta.sequencing.sequencer import OrderedTrack, sequence


def _track(
    path: str,
    bpm: float,
    key_camelot: str,
    intensity: float,
    spectral: dict[str, dict[str, float]],
    phrases: tuple[PhraseRegion, ...],
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


def test_sequence_orders_tracks_once_and_follows_arc_targets():
    tracks = [
        _track(
            path="/music/a.wav",
            bpm=120.0,
            key_camelot="8A",
            intensity=0.10,
            spectral={"outro": {"low": 1.0, "low_mid": 0.0, "high_mid": 0.0, "high": 0.0}},
            phrases=(_phrase("outro"),),
        ),
        _track(
            path="/music/b.wav",
            bpm=122.0,
            key_camelot="8B",
            intensity=0.22,
            spectral={"intro": {"low": 0.0, "low_mid": 1.0, "high_mid": 0.0, "high": 0.0}},
            phrases=(_phrase("intro"),),
        ),
        _track(
            path="/music/c.wav",
            bpm=124.0,
            key_camelot="9B",
            intensity=0.38,
            spectral={"body": {"low": 0.0, "low_mid": 0.0, "high_mid": 1.0, "high": 0.0}},
            phrases=(_phrase("outro"), _phrase("body")),
        ),
        _track(
            path="/music/d.wav",
            bpm=126.0,
            key_camelot="10B",
            intensity=0.55,
            spectral={"intro": {"low": 0.0, "low_mid": 0.0, "high_mid": 0.0, "high": 1.0}},
            phrases=(_phrase("intro"), _phrase("body")),
        ),
        _track(
            path="/music/e.wav",
            bpm=128.0,
            key_camelot="10A",
            intensity=0.74,
            spectral={"outro": {"low": 0.3, "low_mid": 0.2, "high_mid": 0.2, "high": 0.3}},
            phrases=(_phrase("outro"),),
        ),
        _track(
            path="/music/f.wav",
            bpm=130.0,
            key_camelot="11A",
            intensity=0.90,
            spectral={"intro": {"low": 0.25, "low_mid": 0.25, "high_mid": 0.25, "high": 0.25}},
            phrases=(_phrase("intro"),),
        ),
    ]

    result = sequence(tracks)

    assert isinstance(result, tuple)
    ordered, stranded = result
    assert isinstance(ordered, list)
    assert isinstance(stranded, list)
    assert len(ordered) == len(tracks)
    assert [item.position for item in ordered] == list(range(len(tracks)))
    assert all(isinstance(item, OrderedTrack) for item in ordered)

    ordered_paths = [item.track.path for item in ordered]
    assert sorted(ordered_paths) == sorted(track.path for track in tracks)
    assert len(set(ordered_paths)) == len(tracks)

    points = load_profile("experiential")
    expected_targets = [
        arc_target(0.0 if len(tracks) <= 1 else index / (len(tracks) - 1), points)
        for index in range(len(tracks))
    ]
    assert [item.arc_target for item in ordered] == expected_targets

    median_intensity = median(track.intensity for track in tracks)
    assert ordered[0].track.intensity <= median_intensity

    max_intensity = max(track.intensity for track in tracks)
    max_index = next(
        index for index, item in enumerate(ordered) if item.track.intensity == max_intensity
    )
    assert max_index > 0

    assert ordered[0].incoming_score is None
    assert all(item.incoming_score is None or 0.0 <= item.incoming_score <= 100.0 for item in ordered)

    stranded_paths = sorted(track.path for track in stranded)
    expected_stranded_paths = sorted(
        item.track.path
        for item in ordered[1:]
        if item.incoming_score is not None and item.incoming_score < 25.0
    )
    assert stranded_paths == expected_stranded_paths
