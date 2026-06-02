import re

import numba
import numpy as np
import pytest
from numba.core import decorators
from numba.np.ufunc import decorators as ufunc_decorators

from selecta.features.key import detect_key


def _synth_a_minor_tone_bed(sr: int = 22_050, duration_seconds: float = 4.0) -> np.ndarray:
    t = np.linspace(0.0, duration_seconds, int(sr * duration_seconds), endpoint=False)
    envelope = np.hanning(t.size)
    frequencies = (110.0, 130.8128, 164.8138, 220.0, 261.6256, 329.6276)
    amplitudes = (1.0, 0.8, 0.9, 0.65, 0.5, 0.55)

    y = np.zeros_like(t, dtype=np.float64)
    for frequency, amplitude in zip(frequencies, amplitudes, strict=True):
        y += amplitude * np.sin(2 * np.pi * frequency * t)

    peak = float(np.max(np.abs(y), initial=0.0))
    if peak > 0.0:
        y = 0.5 * y / peak

    return y * envelope


def test_detect_key_returns_valid_camelot_for_synthesized_a_minor_bed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sr = 22_050
    y = _synth_a_minor_tone_bed(sr=sr)

    original_jit = decorators.jit
    original_vectorize = ufunc_decorators.vectorize
    original_guvectorize = ufunc_decorators.guvectorize

    def _jit_without_cache(*args, **kwargs):
        patched_kwargs = dict(kwargs)
        patched_kwargs["cache"] = False
        return original_jit(*args, **patched_kwargs)

    def _vectorize_without_cache(*args, **kwargs):
        patched_kwargs = dict(kwargs)
        patched_kwargs["cache"] = False
        return original_vectorize(*args, **patched_kwargs)

    def _guvectorize_without_cache(*args, **kwargs):
        patched_kwargs = dict(kwargs)
        patched_kwargs["cache"] = False
        return original_guvectorize(*args, **patched_kwargs)

    monkeypatch.setattr(numba, "jit", _jit_without_cache)
    monkeypatch.setattr(numba, "vectorize", _vectorize_without_cache)
    monkeypatch.setattr(numba, "guvectorize", _guvectorize_without_cache)
    monkeypatch.setattr(decorators, "jit", _jit_without_cache)
    monkeypatch.setattr(ufunc_decorators, "vectorize", _vectorize_without_cache)
    monkeypatch.setattr(ufunc_decorators, "guvectorize", _guvectorize_without_cache)

    camelot, confidence = detect_key(y, sr)

    assert re.fullmatch(r"(?:[1-9]|1[0-2])[AB]", camelot)
    assert camelot in {"8A", "8B", "7A", "9A"}
    assert confidence > 0.0
    assert confidence <= 1.0
