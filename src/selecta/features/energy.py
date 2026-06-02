"""Energy feature extraction for Selecta."""

from __future__ import annotations

import librosa
import numpy as np


def energy_profile(
    y: "np.ndarray", sr: int, hop_length: int = 512
) -> tuple[tuple[float, ...], float]:
    """Compute a normalized frame-energy curve and a simple intensity scalar.

    The energy curve is the frame RMS normalized to ``[0, 1]`` by its peak value.
    Intensity is the mean of that normalized curve, which keeps the summary easy to
    reason about and stable for downstream phrasing features.
    """

    if y.size == 0:
        return (), 0.0

    rms = librosa.feature.rms(y=np.asarray(y, dtype=float), hop_length=hop_length)[0]
    peak = float(np.max(rms, initial=0.0))

    if peak <= 0.0:
        curve = np.zeros_like(rms, dtype=float)
        return tuple(float(value) for value in curve), 0.0

    curve = np.clip(rms / peak, 0.0, 1.0)
    intensity = float(np.clip(np.mean(curve), 0.0, 1.0))

    return tuple(float(value) for value in curve), intensity
