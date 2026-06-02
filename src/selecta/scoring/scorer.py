"""Pairwise transition scoring for track-to-track mix decisions."""

from __future__ import annotations

from dataclasses import dataclass
from math import fsum

from selecta.features.types import PhraseRegion, TrackFeatures
from selecta.scoring.camelot import mood, relation

_BREAKDOWN_KEYS = ("harmonic", "tempo", "energy", "spectral", "phrase")
_BAND_NAMES = ("low", "low_mid", "high_mid", "high")
_HARMONIC_BASE_SCORES = {
    "same": 1.0,
    "adjacent": 0.85,
    "relative": 0.8,
    "energy_boost": 0.6,
    "distant": 0.2,
}


@dataclass(frozen=True)
class TransitionScore:
    total: float
    breakdown: dict[str, float]


def score(
    a: TrackFeatures,
    b: TrackFeatures,
    arc_target: float,
    weights: dict[str, float],
) -> TransitionScore:
    """Score how well track ``a`` transitions into track ``b``."""

    breakdown = {
        "harmonic": _harmonic_score(a, b, arc_target),
        "tempo": _tempo_score(a, b),
        "energy": _clamp01(1.0 - abs(b.intensity - arc_target)),
        "spectral": _spectral_score(a, b),
        "phrase": _phrase_score(a.phrases, b.phrases),
    }
    normalized_weights = _normalize_weights(weights)
    total = 100.0 * fsum(
        normalized_weights[key] * breakdown[key] for key in _BREAKDOWN_KEYS
    )
    return TransitionScore(total=total, breakdown=breakdown)


def _harmonic_score(a: TrackFeatures, b: TrackFeatures, arc_target: float) -> float:
    score_value = _HARMONIC_BASE_SCORES[relation(a.key_camelot, b.key_camelot)]
    key_mood = mood(a.key_camelot, b.key_camelot)

    if arc_target > a.intensity and key_mood == "lift":
        score_value += 0.1
    elif arc_target < a.intensity and key_mood == "darken":
        score_value += 0.1

    return _clamp01(score_value)


def _tempo_score(a: TrackFeatures, b: TrackFeatures) -> float:
    if a.bpm <= 0:
        return 0.0

    stretch = min(abs((factor * b.bpm) - a.bpm) / a.bpm for factor in (1.0, 0.5, 2.0))
    return _clamp01(1.0 - (stretch / 0.06))


def _spectral_score(a: TrackFeatures, b: TrackFeatures) -> float:
    a_vector = _pick_region_vector(a.spectral, preferred=("outro", "body"))
    b_vector = _pick_region_vector(b.spectral, preferred=("intro", "body"))

    if a_vector is None or b_vector is None:
        return 0.5

    a_total = fsum(a_vector)
    b_total = fsum(b_vector)
    if a_total <= 0.0 or b_total <= 0.0:
        return 0.5

    a_normalized = [value / a_total for value in a_vector]
    b_normalized = [value / b_total for value in b_vector]
    overlap = fsum(min(a_value, b_value) for a_value, b_value in zip(a_normalized, b_normalized))
    return _clamp01(1.0 - overlap)


def _phrase_score(
    a_phrases: tuple[PhraseRegion, ...], b_phrases: tuple[PhraseRegion, ...]
) -> float:
    has_outro = any(phrase.label == "outro" for phrase in a_phrases)
    has_intro = any(phrase.label == "intro" for phrase in b_phrases)

    if has_outro and has_intro:
        return 1.0
    if has_outro or has_intro:
        return 0.6
    return 0.3


def _pick_region_vector(
    spectral: dict[str, dict[str, float]], preferred: tuple[str, ...]
) -> list[float] | None:
    for region in preferred:
        if region in spectral:
            return [float(spectral[region].get(band, 0.0)) for band in _BAND_NAMES]

    for bands in spectral.values():
        return [float(bands.get(band, 0.0)) for band in _BAND_NAMES]

    return None


def _normalize_weights(weights: dict[str, float]) -> dict[str, float]:
    raw_weights = {key: float(weights.get(key, 0.0)) for key in _BREAKDOWN_KEYS}
    total = fsum(raw_weights.values())
    if total <= 0.0:
        equal_weight = 1.0 / len(_BREAKDOWN_KEYS)
        return {key: equal_weight for key in _BREAKDOWN_KEYS}
    return {key: raw_weights[key] / total for key in _BREAKDOWN_KEYS}


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
