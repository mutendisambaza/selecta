import json

from selecta.features.types import Beat, PhraseRegion, TrackFeatures


def test_track_features_json_round_trip():
    track_features = TrackFeatures(
        path="/music/set-opener.wav",
        duration=312.5,
        bpm=128.0,
        bpm_confidence=0.97,
        beats=(
            Beat(time=0.0, is_downbeat=True),
            Beat(time=0.46875, is_downbeat=False),
            Beat(time=0.9375, is_downbeat=False),
        ),
        key_camelot="8A",
        key_confidence=0.88,
        energy_curve=(0.21, 0.35, 0.58, 0.91),
        intensity=0.74,
        spectral={
            "intro": {"low": 0.4, "mid": 0.55, "high": 0.22},
            "drop": {"low": 0.92, "mid": 0.81, "high": 0.67},
        },
        phrases=(
            PhraseRegion(
                start=0.0,
                end=31.5,
                bars=16,
                label="intro",
                confidence=0.89,
            ),
        ),
        title="Set Opener",
        artist="DJ Selecta",
    )

    payload = track_features.to_dict()

    assert TrackFeatures.from_dict(payload) == track_features
    assert json.dumps(payload)
