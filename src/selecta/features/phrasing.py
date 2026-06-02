"""Phrase detection over beat grids and coarse energy curves."""

from __future__ import annotations

from statistics import fmean

from selecta.features.types import Beat, PhraseRegion


PHRASE_BARS = 16
_ALLOWED_LABELS = {"intro", "build", "drop", "breakdown", "outro", "body"}


def detect_phrases(
    beats: tuple[Beat, ...],
    energy_curve: tuple[float, ...],
    duration: float,
) -> tuple[PhraseRegion, ...]:
    """Infer coarse phrase regions from downbeats and energy.

    The detector uses downbeats as bar boundaries and prefers 16-bar phrases.
    When the beat grid does not support at least three stable phrase windows,
    it falls back to four equal-duration regions so the phrase map can still
    represent an intro, body/drop area, and outro.

    Confidence is a single heuristic applied to each region. It decreases when:
    - there are few downbeats relative to the requested 16-bar grouping
    - downbeat spacing is inconsistent, using coefficient of variation
    - the detector had to use the equal-region fallback
    """

    if duration <= 0.0:
        return tuple()

    downbeats = _sorted_downbeats(beats, duration)
    boundaries, bars_per_region, used_fallback = _phrase_boundaries(downbeats, duration)
    if len(boundaries) < 2:
        return tuple()

    region_energies = tuple(
        _mean_energy(energy_curve, duration, start, end)
        for start, end in zip(boundaries, boundaries[1:])
    )
    confidence = _phrase_confidence(downbeats, used_fallback)
    labels = _label_regions(region_energies)

    regions = []
    for index, (start, end) in enumerate(zip(boundaries, boundaries[1:])):
        label = labels[index]
        if label not in _ALLOWED_LABELS:
            label = "body"
        regions.append(
            PhraseRegion(
                start=start,
                end=end,
                bars=bars_per_region[index],
                label=label,
                confidence=confidence,
            )
        )
    return tuple(regions)


def _sorted_downbeats(beats: tuple[Beat, ...], duration: float) -> tuple[float, ...]:
    times = sorted(
        beat.time
        for beat in beats
        if beat.is_downbeat and 0.0 <= beat.time < duration + 1e-9
    )
    if not times or abs(times[0]) > 1e-6:
        times.insert(0, 0.0)

    deduped: list[float] = []
    for time in times:
        if not deduped or abs(time - deduped[-1]) > 1e-6:
            deduped.append(min(max(time, 0.0), duration))
    return tuple(deduped)


def _phrase_boundaries(
    downbeats: tuple[float, ...], duration: float
) -> tuple[tuple[float, ...], tuple[int, ...], bool]:
    if len(downbeats) >= PHRASE_BARS + 1:
        phrase_count = (len(downbeats) - 1) // PHRASE_BARS
    else:
        phrase_count = 0

    if phrase_count >= 3:
        boundaries = [downbeats[0]]
        bars = []
        for phrase_index in range(1, phrase_count):
            boundary = downbeats[phrase_index * PHRASE_BARS]
            boundaries.append(boundary)
            bars.append(PHRASE_BARS)
        boundaries.append(duration)

        remaining_bars = max(len(downbeats) - 1 - PHRASE_BARS * (phrase_count - 1), 1)
        bars.append(remaining_bars)
        return tuple(boundaries), tuple(bars), False

    fallback_boundaries = [0.0, duration * 0.25, duration * 0.5, duration * 0.75, duration]
    aligned = [_align_to_downbeat(boundary, downbeats, duration) for boundary in fallback_boundaries]
    aligned[0] = 0.0
    aligned[-1] = duration

    monotonic = [aligned[0]]
    for boundary in aligned[1:-1]:
        if boundary <= monotonic[-1]:
            continue
        monotonic.append(boundary)
    monotonic.append(duration)

    bars = []
    for start, end in zip(monotonic, monotonic[1:]):
        bars.append(_count_bars_between(downbeats, start, end))
    return tuple(monotonic), tuple(bars), True


def _align_to_downbeat(boundary: float, downbeats: tuple[float, ...], duration: float) -> float:
    if not downbeats:
        return min(max(boundary, 0.0), duration)
    return min(downbeats, key=lambda time: (abs(time - boundary), time))


def _count_bars_between(downbeats: tuple[float, ...], start: float, end: float) -> int:
    covered = sum(1 for time in downbeats if start <= time < end - 1e-9)
    return max(covered, 1)


def _mean_energy(
    energy_curve: tuple[float, ...], duration: float, start: float, end: float
) -> float:
    if not energy_curve or duration <= 0.0 or end <= start:
        return 0.0

    start_ratio = start / duration
    end_ratio = end / duration
    values = [
        sample
        for index, sample in enumerate(energy_curve)
        if start_ratio <= index / len(energy_curve) < end_ratio
    ]
    return fmean(values) if values else 0.0


def _phrase_confidence(downbeats: tuple[float, ...], used_fallback: bool) -> float:
    if len(downbeats) < 2:
        return 0.0

    intervals = [right - left for left, right in zip(downbeats, downbeats[1:]) if right > left]
    if not intervals:
        return 0.0

    mean_interval = fmean(intervals)
    variance = fmean((interval - mean_interval) ** 2 for interval in intervals)
    spacing_cv = (variance ** 0.5) / mean_interval if mean_interval > 0.0 else 1.0
    spacing_score = max(0.0, 1.0 - min(spacing_cv, 1.0))
    count_score = min(len(downbeats) / (PHRASE_BARS * 4), 1.0)
    fallback_penalty = 0.2 if used_fallback else 0.0
    confidence = 0.25 + 0.45 * spacing_score + 0.3 * count_score - fallback_penalty
    return max(0.0, min(confidence, 1.0))


def _label_regions(energies: tuple[float, ...]) -> tuple[str, ...]:
    count = len(energies)
    if count == 0:
        return tuple()
    if count == 1:
        return ("intro",)

    labels = ["body"] * count
    labels[0] = "intro"
    labels[-1] = "outro"

    interior = range(1, count - 1)
    if interior:
        drop_index = max(interior, key=lambda index: energies[index], default=None)
    else:
        drop_index = None

    if drop_index is None and count > 2:
        drop_index = max(range(count), key=lambda index: energies[index])

    if drop_index is not None and drop_index not in (0, count - 1):
        labels[drop_index] = "drop"

        breakdown_index = drop_index + 1
        if breakdown_index < count - 1:
            drop_energy = energies[drop_index]
            next_energy = energies[breakdown_index]
            previous_energy = energies[breakdown_index - 1]
            if (
                next_energy < previous_energy
                and next_energy <= drop_energy * 0.8
                and next_energy <= fmean(energies)
            ):
                labels[breakdown_index] = "breakdown"

    for index in range(1, count - 1):
        if labels[index] != "body":
            continue
        if energies[index] > energies[index - 1] + 0.05:
            labels[index] = "build"

    return tuple(labels)
