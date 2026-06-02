"""Energy feature extraction for Selecta."""

from __future__ import annotations

import librosa
import numpy as np

# Spectral centroid reference: tracks whose mean centroid meets or exceeds this
# frequency saturate at 1.0 for the brightness component. Tune per-genre.
_CENTROID_REFERENCE_HZ = 4000.0

# Beat HF fraction reference: proportion of beat-frame power above _HF_CUTOFF_HZ
# that maps to 1.0. 0.50 means "half of all beat energy in the HF band = max score".
_BEAT_HF_REFERENCE = 0.50

# Lower bound of the "high-frequency" band used for beat-sync HF fraction.
_HF_CUTOFF_HZ = 4000.0

# Composite weights (must sum to 1.0).
_CENTROID_WEIGHT = 0.55
_BEAT_HF_WEIGHT = 0.45

_N_FFT = 2048
_MIN_BEATS_FOR_SYNC = 4  # fall back to frame-average when too few beats detected


def energy_profile(
    y: "np.ndarray", sr: int, hop_length: int = 512
) -> tuple[tuple[float, ...], float]:
    """Compute a normalized frame-energy curve and cross-track intensity scalar.

    The energy curve is the frame RMS normalized to [0, 1] by the track's own peak
    (relative shape only — not cross-track comparable).

    The intensity scalar is a spectral-brightness composite robust to loudness-
    normalized / heavily mastered material:

      1. Spectral centroid (mean across frames) clipped to a fixed reference
         frequency. Captures spectral tilt without reference to absolute level.
      2. Beat-synchronized HF fraction — proportion of power above _HF_CUTOFF_HZ
         at detected beat positions. Falls back to frame-average if <4 beats found.

    Both are spectral *shape* measures, so they remain discriminating even when
    mean RMS is uniform across a uniformly-mastered crate (e.g. amapiano).
    The sequencer applies a final rank/percentile calibration on top.
    """
    samples = np.asarray(y, dtype=float)

    if samples.size == 0:
        return (), 0.0

    rms = _rms(samples, hop_length=hop_length)
    peak = float(np.max(rms, initial=0.0))

    if peak <= 0.0:
        curve = np.zeros_like(rms, dtype=float)
        return tuple(float(v) for v in curve), 0.0

    curve = np.clip(rms / peak, 0.0, 1.0)

    stft = librosa.stft(samples, n_fft=_N_FFT, hop_length=hop_length)
    S_mag = np.abs(stft)
    S_power = S_mag ** 2

    freqs = librosa.fft_frequencies(sr=sr, n_fft=_N_FFT)
    hf_mask = freqs >= _HF_CUTOFF_HZ

    # Feature 1: mean spectral centroid, normalized to reference frequency
    centroid = librosa.feature.spectral_centroid(S=S_mag, sr=sr, n_fft=_N_FFT)[0]
    centroid_mean = float(np.nan_to_num(np.mean(centroid), nan=0.0))
    spectral_brightness = float(np.clip(centroid_mean / _CENTROID_REFERENCE_HZ, 0.0, 1.0))

    # Feature 2: beat-synchronized HF fraction
    try:
        _, beat_frames = librosa.beat.beat_track(y=samples, sr=sr, hop_length=hop_length)
    except Exception:
        beat_frames = np.array([], dtype=int)

    n_frames = S_power.shape[1]
    valid_beats = beat_frames[beat_frames < n_frames] if len(beat_frames) > 0 else np.array([], dtype=int)

    if len(valid_beats) >= _MIN_BEATS_FOR_SYNC:
        beat_hf = S_power[np.ix_(hf_mask, valid_beats)].sum(axis=0)
        beat_total = S_power[:, valid_beats].sum(axis=0) + 1e-10
        beat_hf_frac = float(np.mean(beat_hf / beat_total))
    else:
        hf_total = float(S_power[hf_mask, :].sum())
        beat_hf_frac = hf_total / (float(S_power.sum()) + 1e-10)

    beat_hf_score = float(np.clip(beat_hf_frac / _BEAT_HF_REFERENCE, 0.0, 1.0))

    intensity = float(
        np.clip(
            _CENTROID_WEIGHT * spectral_brightness + _BEAT_HF_WEIGHT * beat_hf_score,
            0.0,
            1.0,
        )
    )

    return tuple(float(v) for v in curve), intensity


def _rms(y: "np.ndarray", hop_length: int) -> "np.ndarray":
    try:
        return librosa.feature.rms(y=y, hop_length=hop_length)[0]
    except RuntimeError as exc:
        if "cannot cache function" not in str(exc):
            raise
        return _numpy_rms(y, hop_length=hop_length)


def _numpy_rms(y: "np.ndarray", hop_length: int, frame_length: int = 2048) -> "np.ndarray":
    if y.size == 0:
        return np.empty(0, dtype=float)

    padded = np.pad(y, (frame_length // 2, frame_length // 2), mode="constant")
    if padded.size < frame_length:
        padded = np.pad(padded, (0, frame_length - padded.size), mode="constant")

    frame_count = 1 + (padded.size - frame_length) // hop_length
    frame_starts = hop_length * np.arange(frame_count)
    frames = np.stack(
        [padded[start : start + frame_length] for start in frame_starts],
        axis=0,
    )
    return np.sqrt(np.mean(frames * frames, axis=1))
