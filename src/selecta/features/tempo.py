"""Tempo and beat-grid extraction utilities."""

from __future__ import annotations

import librosa
import numpy as np

from selecta.features.types import Beat


def _compute_confidence(beat_times: np.ndarray) -> float:
    """Estimate confidence from beat-interval regularity."""

    if beat_times.size < 2:
        return 0.0

    intervals = np.diff(beat_times)
    mean_interval = float(np.mean(intervals))
    if mean_interval <= 0.0:
        return 0.0

    interval_std = float(np.std(intervals))
    confidence = 1.0 - min(1.0, interval_std / mean_interval)
    return float(np.clip(confidence, 0.0, 1.0))


def detect_tempo(y: np.ndarray, sr: int) -> tuple[float, float, tuple[Beat, ...]]:
    """Detect tempo, confidence, and a beat grid with 4/4 downbeats."""

    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)

    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    bpm = float(np.atleast_1d(tempo)[0]) if np.size(tempo) else 0.0
    if beat_times.size >= 2:
        mean_interval = float(np.mean(np.diff(beat_times)))
        if mean_interval > 0.0:
            bpm = 60.0 / mean_interval
    confidence = _compute_confidence(beat_times)

    beats = tuple(
        Beat(time=float(beat_time), is_downbeat=index % 4 == 0)
        for index, beat_time in enumerate(beat_times)
    )

    return bpm, confidence, beats
