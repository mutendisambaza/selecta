"""Key detection utilities that return Camelot wheel notation."""

from __future__ import annotations

import librosa
import numpy as np

_KS_MAJOR_PROFILE = np.array(
    [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88],
    dtype=float,
)
_KS_MINOR_PROFILE = np.array(
    [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17],
    dtype=float,
)

_CAM_MAJOR = {
    0: "8B",
    7: "9B",
    2: "10B",
    9: "11B",
    4: "12B",
    11: "1B",
    6: "2B",
    1: "3B",
    8: "4B",
    3: "5B",
    10: "6B",
    5: "7B",
}
_CAM_MINOR = {
    9: "8A",
    4: "9A",
    11: "10A",
    6: "11A",
    1: "12A",
    8: "1A",
    3: "2A",
    10: "3A",
    5: "4A",
    0: "5A",
    7: "6A",
    2: "7A",
}


def _correlate_pitch_class_profile(
    pitch_class_profile: np.ndarray, template: np.ndarray
) -> np.ndarray:
    """Return correlation scores for every tonic rotation of one key template."""

    centered_profile = pitch_class_profile - np.mean(pitch_class_profile)
    centered_template = template - np.mean(template)

    profile_norm = float(np.linalg.norm(centered_profile))
    template_norm = float(np.linalg.norm(centered_template))
    if profile_norm <= 0.0 or template_norm <= 0.0:
        return np.zeros(12, dtype=float)

    scores = np.empty(12, dtype=float)
    denominator = profile_norm * template_norm
    for tonic in range(12):
        rotated = np.roll(centered_template, tonic)
        scores[tonic] = float(np.dot(centered_profile, rotated) / denominator)
    return scores


def detect_key(y: "np.ndarray", sr: int) -> tuple[str, float]:
    """Detect the musical key and return a Camelot code with confidence."""

    samples = np.asarray(y, dtype=float)
    if samples.size == 0 or not np.any(np.abs(samples) > 0.0):
        return "8A", 0.0

    chroma = librosa.feature.chroma_cqt(y=samples, sr=sr)
    pitch_class_profile = np.mean(chroma, axis=1)
    if pitch_class_profile.size != 12 or float(np.sum(pitch_class_profile)) <= 0.0:
        return "8A", 0.0

    major_scores = _correlate_pitch_class_profile(pitch_class_profile, _KS_MAJOR_PROFILE)
    minor_scores = _correlate_pitch_class_profile(pitch_class_profile, _KS_MINOR_PROFILE)

    all_scores = np.concatenate((major_scores, minor_scores))
    best_index = int(np.argmax(all_scores))
    best_score = float(all_scores[best_index])

    is_major = best_index < 12
    tonic = best_index if is_major else best_index - 12
    camelot = _CAM_MAJOR[tonic] if is_major else _CAM_MINOR[tonic]

    score_span = float(np.max(all_scores) - np.min(all_scores))
    if score_span <= 0.0:
        confidence = 0.0
    else:
        confidence = float(np.clip((best_score - float(np.mean(all_scores))) / score_span, 0.0, 1.0))

    return camelot, confidence
