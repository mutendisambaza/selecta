from selecta.features.types import Beat, PhraseRegion, TrackFeatures
from selecta.store.db import FeatureStore


def make_track_features(path: str) -> TrackFeatures:
    return TrackFeatures(
        path=path,
        duration=312.5,
        bpm=128.0,
        bpm_confidence=0.97,
        beats=(
            Beat(time=0.0, is_downbeat=True),
            Beat(time=0.46875, is_downbeat=False),
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


def test_feature_store_round_trip_and_all(tmp_path):
    audio_path = tmp_path / "track.wav"
    audio_path.write_bytes(b"first pass")
    store = FeatureStore(str(tmp_path / "features.sqlite3"))
    track_features = make_track_features(str(audio_path))

    try:
        store.upsert(track_features)

        assert store.get(track_features.path) == track_features
        assert store.all() == [track_features]
    finally:
        store.close()


def test_feature_store_needs_analysis_tracks_content_changes(tmp_path):
    audio_path = tmp_path / "track.wav"
    audio_path.write_bytes(b"first pass")
    store = FeatureStore(str(tmp_path / "features.sqlite3"))
    track_features = make_track_features(str(audio_path))

    try:
        assert store.needs_analysis(track_features.path) is True

        store.upsert(track_features)

        assert store.needs_analysis(track_features.path) is False

        audio_path.write_bytes(b"second pass")

        assert store.needs_analysis(track_features.path) is True
    finally:
        store.close()
