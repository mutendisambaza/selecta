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

# BPM is estimated from a windowed segment of the audio to avoid intro/outro and
# breakdown sections that confuse the beat tracker (observed in amapiano/gqom).
_BPM_SKIP_START_SEC = 15.0     # skip this many seconds from the beginning
_BPM_WINDOW_SEC     = 90.0     # analyse at most this many seconds for BPM


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


def _estimate_bpm(y: np.ndarray, sr: int) -> float:
    """Estimate BPM from a middle section of the audio.

    For tracks long enough to window (> skip + 30 s), uses onset-strength
    autocorrelation on a [skip, skip+window] segment — more robust to intros,
    outros, and breakdown sections (observed in amapiano/gqom).

    For short tracks falls back to the beat interval mean from the full signal,
    which is accurate for uniform synthetic material.
    """
    skip = int(_BPM_SKIP_START_SEC * sr)
    window = int(_BPM_WINDOW_SEC * sr)
    min_windowed = skip + int(30 * sr)  # need at least 30 s after the skip

    if len(y) >= min_windowed:
        end = min(len(y), skip + window)
        y_seg = y[skip:end]
        onset_env = librosa.onset.onset_strength(y=y_seg, sr=sr)
        tempo = librosa.feature.tempo(onset_envelope=onset_env, sr=sr)
        return float(np.atleast_1d(tempo)[0])

    # Short track: use beat interval mean on the full signal (reliable for
    # uniform content like synthesized click tracks or short stems).
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    if beat_times.size >= 2:
        return 60.0 / float(np.mean(np.diff(beat_times)))
    return float(np.atleast_1d(tempo)[0])


def detect_tempo(y: np.ndarray, sr: int) -> tuple[float, float, tuple[Beat, ...]]:
    """Detect tempo, confidence, and a beat grid with 4/4 downbeats."""

    # Beat grid from the full track (needed for phrase detection).
    _, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)

    # BPM scalar from a windowed middle segment — more robust than using the full
    # track's beat interval mean, which can be corrupted by breakdowns or outros.
    bpm = _estimate_bpm(y, sr)
    bpm = _fold_bpm(bpm)

    confidence = _compute_confidence(beat_times)

    beats = tuple(
        Beat(time=float(beat_time), is_downbeat=index % 4 == 0)
        for index, beat_time in enumerate(beat_times)
    )

    return bpm, confidence, beats
