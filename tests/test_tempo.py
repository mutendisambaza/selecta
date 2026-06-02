from __future__ import annotations

import numpy as np

from selecta.features.tempo import detect_tempo


def _synth_click_track(
    bpm: float = 120.0,
    duration_seconds: float = 8.0,
    sr: int = 22_050,
) -> np.ndarray:
    samples = np.zeros(int(duration_seconds * sr), dtype=np.float32)
    beat_interval_seconds = 60.0 / bpm
    click_length = max(1, int(0.01 * sr))

    for beat_time in np.arange(0.0, duration_seconds, beat_interval_seconds):
        click_start = int(round(beat_time * sr))
        click_end = min(click_start + click_length, samples.size)
        samples[click_start:click_end] = 1.0

    return samples


def test_detect_tempo_on_synthesized_click_track() -> None:
    sr = 22_050
    y = _synth_click_track(sr=sr)

    bpm, confidence, beats = detect_tempo(y, sr)

    assert abs(bpm - 120.0) <= 2.0
    assert 0.0 <= confidence <= 1.0
    assert beats
    assert any(beat.is_downbeat for beat in beats)

    beat_times = np.array([beat.time for beat in beats], dtype=np.float64)
    intervals = np.diff(beat_times)

    assert intervals.size > 0
    assert np.allclose(intervals, 0.5, atol=0.07)


def test_fold_bpm_corrects_half_time():
    from selecta.features.tempo import _fold_bpm

    assert abs(_fold_bpm(63.0) - 126.0) < 1e-6   # amapiano half-time -> doubled
    assert abs(_fold_bpm(124.0) - 124.0) < 1e-6  # in-range unchanged
    assert abs(_fold_bpm(200.0) - 100.0) < 1e-6  # too-fast -> halved
    assert _fold_bpm(0.0) == 0.0
