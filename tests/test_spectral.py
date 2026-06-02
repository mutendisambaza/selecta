import numpy as np

from selecta.features.spectral import band_profile


def test_band_profile_low_sine_is_low_dominant_and_normalized():
    sr = 22_050
    duration_seconds = 1.0
    t = np.linspace(0.0, duration_seconds, int(sr * duration_seconds), endpoint=False)
    y = np.sin(2 * np.pi * 80.0 * t)

    profile = band_profile(y, sr, [(0.0, duration_seconds, "section_a")])
    bands = profile["section_a"]

    assert max(bands, key=bands.get) == "low"
    assert abs(sum(bands.values()) - 1.0) <= 1e-6


def test_band_profile_high_sine_is_high_dominant_and_normalized():
    sr = 22_050
    duration_seconds = 1.0
    t = np.linspace(0.0, duration_seconds, int(sr * duration_seconds), endpoint=False)
    y = np.sin(2 * np.pi * 10_000.0 * t)

    profile = band_profile(y, sr, [(0.0, duration_seconds, "section_b")])
    bands = profile["section_b"]

    assert max(bands, key=bands.get) == "high"
    assert abs(sum(bands.values()) - 1.0) <= 1e-6


def test_band_profile_silent_region_returns_zeroes():
    y = np.zeros(22_050, dtype=float)

    profile = band_profile(y, 22_050, [(0.0, 1.0, "silent")])

    assert profile["silent"] == {
        "low": 0.0,
        "low_mid": 0.0,
        "high_mid": 0.0,
        "high": 0.0,
    }


def test_band_profile_last_duplicate_label_wins():
    sr = 22_050
    duration_seconds = 2.0
    t = np.linspace(0.0, duration_seconds, int(sr * duration_seconds), endpoint=False)
    first_half = np.sin(2 * np.pi * 80.0 * t[: sr])
    second_half = np.sin(2 * np.pi * 10_000.0 * t[: sr])
    y = np.concatenate((first_half, second_half))

    profile = band_profile(
        y,
        sr,
        [
            (0.0, 1.0, "shared"),
            (1.0, 2.0, "shared"),
        ],
    )
    bands = profile["shared"]

    assert max(bands, key=bands.get) == "high"
    assert abs(sum(bands.values()) - 1.0) <= 1e-6
