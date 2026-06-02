"""Top-level feature extraction orchestration for Selecta."""

from __future__ import annotations

from pathlib import Path

import librosa
from mutagen import File as MutagenFile

from selecta.features.energy import energy_profile
from selecta.features.key import detect_key
from selecta.features.phrasing import detect_phrases
from selecta.features.spectral import band_profile
from selecta.features.tempo import detect_tempo
from selecta.features.types import TrackFeatures


class ExtractionError(Exception):
    """Raised when a track cannot be decoded into features."""


def extract(path: str) -> TrackFeatures:
    """Decode audio, run sub-extractors, and assemble ``TrackFeatures``."""

    try:
        y, sr = librosa.load(path, sr=None, mono=True)
    except Exception as exc:
        raise ExtractionError(f"Failed to load audio from {path}") from exc

    if len(y) == 0 or sr <= 0:
        raise ExtractionError(f"Failed to load audio from {path}: empty audio")

    duration = float(len(y) / sr)
    bpm, bpm_confidence, beats = detect_tempo(y, sr)
    energy_curve, intensity = energy_profile(y, sr)
    phrases = detect_phrases(beats, energy_curve, duration)
    regions = [(phrase.start, phrase.end, phrase.label) for phrase in phrases]
    spectral = band_profile(y, sr, regions) if regions else {}
    key_camelot, key_confidence = detect_key(y, sr)
    title, artist = _read_tags(path)

    return TrackFeatures(
        path=path,
        duration=duration,
        bpm=bpm,
        bpm_confidence=bpm_confidence,
        beats=beats,
        key_camelot=key_camelot,
        key_confidence=key_confidence,
        energy_curve=energy_curve,
        intensity=intensity,
        spectral=spectral,
        phrases=phrases,
        title=title,
        artist=artist,
    )


def _read_tags(path: str) -> tuple[str | None, str | None]:
    fallback_title = Path(path).stem

    try:
        tags = MutagenFile(path, easy=True)
        if tags is None:
            return fallback_title, None

        title = tags.get("title", [None])[0]
        artist = tags.get("artist", [None])[0]
        return title or fallback_title, artist
    except Exception:
        return fallback_title, None
