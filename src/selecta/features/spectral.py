"""Spectral band feature extraction for Selecta."""

from __future__ import annotations

import librosa
import numpy as np


_BAND_NAMES = ("low", "low_mid", "high_mid", "high")


def band_profile(
    y: "np.ndarray",
    sr: int,
    regions: list[tuple[float, float, str]],
) -> dict[str, dict[str, float]]:
    """Compute per-region spectral energy fractions across four broad bands.

    Regions are provided as ``(start_sec, end_sec, label)`` tuples. If multiple
    regions share the same label, the last region wins in the returned mapping.
    Silent or empty regions return zeroes for every band.
    """

    samples = np.asarray(y, dtype=float)
    profile: dict[str, dict[str, float]] = {}

    if sr <= 0:
        raise ValueError("sr must be positive")

    for start_sec, end_sec, label in regions:
        start_sample = max(0, int(round(start_sec * sr)))
        end_sample = min(samples.size, int(round(end_sec * sr)))

        if end_sample <= start_sample:
            profile[label] = _empty_band_map()
            continue

        region = samples[start_sample:end_sample]
        if region.size == 0 or not np.any(region):
            profile[label] = _empty_band_map()
            continue

        magnitude, n_fft = _stft_magnitude(region)
        power = magnitude**2
        if power.size == 0:
            profile[label] = _empty_band_map()
            continue

        frequencies = _fft_frequencies(sr=sr, n_fft=n_fft)
        band_energies = {
            "low": float(power[_band_mask(frequencies, 20.0, 250.0)].sum()),
            "low_mid": float(power[_band_mask(frequencies, 250.0, 2000.0)].sum()),
            "high_mid": float(power[_band_mask(frequencies, 2000.0, 6000.0)].sum()),
            "high": float(power[_band_mask(frequencies, 6000.0, sr / 2)].sum()),
        }

        total_energy = sum(band_energies.values())
        if total_energy <= 0.0:
            profile[label] = _empty_band_map()
            continue

        profile[label] = {
            band: float(band_energies[band] / total_energy) for band in _BAND_NAMES
        }

    return profile


def _band_mask(
    frequencies: "np.ndarray", lower_hz: float, upper_hz: float
) -> "np.ndarray":
    if upper_hz <= lower_hz:
        return np.zeros_like(frequencies, dtype=bool)

    if upper_hz == frequencies[-1]:
        return (frequencies >= lower_hz) & (frequencies <= upper_hz)

    return (frequencies >= lower_hz) & (frequencies < upper_hz)


def _empty_band_map() -> dict[str, float]:
    return {band: 0.0 for band in _BAND_NAMES}


def _fft_frequencies(sr: int, n_fft: int) -> "np.ndarray":
    try:
        return librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    except RuntimeError as exc:
        if "cannot cache function" not in str(exc):
            raise
        return np.fft.rfftfreq(n_fft, d=1.0 / sr)


def _stft_magnitude(region: "np.ndarray", n_fft: int = 2048) -> tuple["np.ndarray", int]:
    try:
        return np.abs(librosa.stft(region, n_fft=n_fft)), n_fft
    except RuntimeError as exc:
        if "cannot cache function" not in str(exc):
            raise
        return _numpy_stft_magnitude(region, n_fft=n_fft), n_fft


def _numpy_stft_magnitude(region: "np.ndarray", n_fft: int) -> "np.ndarray":
    if region.size == 0:
        return np.empty((n_fft // 2 + 1, 0), dtype=float)

    hop_length = n_fft // 4
    padded = np.pad(region, (n_fft // 2, n_fft // 2), mode="constant")

    if padded.size < n_fft:
        padded = np.pad(padded, (0, n_fft - padded.size), mode="constant")

    frame_count = 1 + (padded.size - n_fft) // hop_length
    frame_starts = hop_length * np.arange(frame_count)
    frames = np.stack([padded[start : start + n_fft] for start in frame_starts], axis=1)
    window = np.hanning(n_fft)[:, np.newaxis]

    return np.abs(np.fft.rfft(frames * window, axis=0))
