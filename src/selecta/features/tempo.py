"""Tempo and beat-grid extraction utilities."""

from __future__ import annotations

import librosa
import numpy as np

from selecta.features.types import Beat

# Fold octave/half-time detection errors into a typical dance range. The target
# genres (house, tech, Afrotech, gqom, amapiano) sit ~110-130 BPM, so librosa's
# common half-time picks (e.g. amapiano detected at ~60) get doubled back up.
_BPM_FOLD_MIN = 90.0
_BPM_FOLD_MAX = 180.0


def _fold_bpm(bpm: float) -> float:
    """Fold a BPM into [_BPM_FOLD_MIN, _BPM_FOLD_MAX) by octaves (x2 / /2)."""
    if bpm <= 0.0:
        return bpm
    while bpm < _BPM_FOLD_MIN:
        bpm *= 2.0
    while bpm >= _BPM_FOLD_MAX:
        bpm /= 2.0
    return bpm


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
    bpm = _fold_bpm(bpm)

    beats = tuple(
        Beat(time=float(beat_time), is_downbeat=index % 4 == 0)
        for index, beat_time in enumerate(beat_times)
    )

    return bpm, confidence, beats
