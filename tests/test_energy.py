import numpy as np

from selecta.features.energy import energy_profile


def test_energy_profile_ramp_signal_increases_and_is_normalized():
    sr = 22_050
    duration_seconds = 4
    sample_count = sr * duration_seconds
    t = np.linspace(0.0, duration_seconds, sample_count, endpoint=False)
    amplitude = np.linspace(0.001, 1.0, sample_count)
    y = amplitude * np.sin(2 * np.pi * 220.0 * t)

    curve, intensity = energy_profile(y, sr=sr)
    curve_array = np.asarray(curve)
    midpoint = len(curve_array) // 2

    assert curve
    assert np.all(curve_array >= 0.0)
    assert np.all(curve_array <= 1.0)
    assert np.mean(curve_array[midpoint:]) > np.mean(curve_array[:midpoint])
    assert 0.0 < intensity <= 1.0


def test_energy_profile_loud_dense_signal_scores_higher_than_quiet_sparse_signal():
    sr = 22_050
    duration_seconds = 4
    sample_count = sr * duration_seconds
    rng = np.random.default_rng(7)

    loud_dense = 0.35 * rng.standard_normal(sample_count)

    t = np.linspace(0.0, duration_seconds, sample_count, endpoint=False)
    quiet_sparse = 0.03 * np.sin(2 * np.pi * 220.0 * t)
    click_positions = np.array([0.5, 2.5]) * sr
    click_indices = click_positions.astype(int)
    quiet_sparse[click_indices] += 0.2

    _, loud_dense_intensity = energy_profile(loud_dense, sr=sr)
    _, quiet_sparse_intensity = energy_profile(quiet_sparse, sr=sr)

    assert loud_dense_intensity > quiet_sparse_intensity + 0.15


def test_energy_profile_silence_returns_zero_curve_and_intensity():
    y = np.zeros(22_050, dtype=float)

    curve, intensity = energy_profile(y, sr=22_050)

    assert curve
    assert all(value == 0.0 for value in curve)
    assert intensity == 0.0
