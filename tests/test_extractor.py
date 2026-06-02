from __future__ import annotations

import re

import numba
import numpy as np
import pytest
import soundfile as sf
from numba.core import decorators
from numba.np.ufunc import decorators as ufunc_decorators

from selecta.features.extractor import ExtractionError, extract
from selecta.features.types import TrackFeatures


def _synth_track(sr: int = 22_050, duration_seconds: float = 12.0) -> np.ndarray:
    t = np.linspace(0.0, duration_seconds, int(sr * duration_seconds), endpoint=False)
    y = np.zeros_like(t, dtype=np.float64)

    beat_interval = 0.5
    kick_length = max(1, int(0.08 * sr))
    kick_env = np.exp(-np.linspace(0.0, 7.0, kick_length))
    kick_wave = np.sin(2.0 * np.pi * 55.0 * np.arange(kick_length) / sr) * kick_env

    for beat_time in np.arange(0.0, duration_seconds, beat_interval):
        start = int(round(beat_time * sr))
        end = min(start + kick_length, y.size)
        y[start:end] += 0.9 * kick_wave[: end - start]

    chord = (
        0.35 * np.sin(2.0 * np.pi * 220.0 * t)
        + 0.28 * np.sin(2.0 * np.pi * 261.6256 * t)
        + 0.32 * np.sin(2.0 * np.pi * 329.6276 * t)
    )
    modulation = 0.55 + 0.45 * np.sin(2.0 * np.pi * 0.125 * t) ** 2
    y += chord * modulation

    peak = float(np.max(np.abs(y), initial=0.0))
    if peak > 0.0:
        y = 0.8 * y / peak

    return y.astype(np.float32)


def _disable_numba_cache(monkeypatch: pytest.MonkeyPatch) -> None:
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


def test_extract_returns_track_features_for_synthesized_audio(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _disable_numba_cache(monkeypatch)

    path = tmp_path / "club-loop.wav"
    sf.write(path, _synth_track(), 22_050)

    features = extract(str(path))

    assert isinstance(features, TrackFeatures)
    assert features.path == str(path)
    assert features.duration > 0.0
    assert features.bpm > 0.0
    assert re.fullmatch(r"(?:[1-9]|1[0-2])[AB]", features.key_camelot)
    assert features.energy_curve
    assert len(features.phrases) >= 1
    assert features.title == "club-loop"
    assert features.artist is None


def test_extract_raises_on_invalid_audio_file(tmp_path) -> None:
    path = tmp_path / "broken.wav"
    path.write_bytes(b"not audio")

    with pytest.raises(ExtractionError):
        extract(str(path))
