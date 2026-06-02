"""Energy feature extraction for Selecta."""

from __future__ import annotations

import librosa
import numpy as np

_LOUDNESS_REFERENCE_RMS = 0.1
_DENSITY_REFERENCE_ONSET_STRENGTH = 4.0
_LOUDNESS_WEIGHT = 0.5
_DENSITY_WEIGHT = 0.5


def energy_profile(
    y: "np.ndarray", sr: int, hop_length: int = 512
) -> tuple[tuple[float, ...], float]:
    """Compute a normalized frame-energy curve and cross-track intensity scalar.

    The energy curve is the frame RMS normalized to ``[0, 1]`` by its peak value.
    Intensity is a fixed-reference blend of absolute loudness and onset density so
    values remain comparable across tracks instead of collapsing around each track's
    own peak level.
    """

    samples = np.asarray(y, dtype=float)

    if samples.size == 0:
        return (), 0.0

    rms = _rms(samples, hop_length=hop_length)
    peak = float(np.max(rms, initial=0.0))

    if peak <= 0.0:
        curve = np.zeros_like(rms, dtype=float)
        return tuple(float(value) for value in curve), 0.0

    curve = np.clip(rms / peak, 0.0, 1.0)
    loudness = float(
        np.clip(np.mean(rms) / _LOUDNESS_REFERENCE_RMS, 0.0, 1.0)
    )
    density = float(
        np.clip(
            np.mean(_onset_strength(samples, sr=sr)) / _DENSITY_REFERENCE_ONSET_STRENGTH,
            0.0,
            1.0,
        )
    )
    intensity = float(
        np.clip(
            (_LOUDNESS_WEIGHT * loudness) + (_DENSITY_WEIGHT * density),
            0.0,
            1.0,
        )
    )

    return tuple(float(value) for value in curve), intensity


def _rms(y: "np.ndarray", hop_length: int) -> "np.ndarray":
    try:
        return librosa.feature.rms(y=y, hop_length=hop_length)[0]
    except RuntimeError as exc:
        if "cannot cache function" not in str(exc):
            raise
        return _numpy_rms(y, hop_length=hop_length)


def _onset_strength(y: "np.ndarray", sr: int) -> "np.ndarray":
    try:
        return librosa.onset.onset_strength(y=y, sr=sr)
    except RuntimeError as exc:
        if "cannot cache function" not in str(exc):
            raise
        return _numpy_onset_strength(y, sr=sr)


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


def _numpy_onset_strength(
    y: "np.ndarray", sr: int, hop_length: int = 512, frame_length: int = 2048
) -> "np.ndarray":
    if y.size == 0:
        return np.empty(0, dtype=float)

    rms = _numpy_rms(y, hop_length=hop_length, frame_length=frame_length)
    if rms.size == 0:
        return np.empty(0, dtype=float)

    delta = np.diff(rms, prepend=rms[:1])
    return np.maximum(delta, 0.0)
