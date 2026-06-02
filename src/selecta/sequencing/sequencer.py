"""Journey-based track ordering for Selecta."""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import fsum

from selecta.features.types import TrackFeatures
from selecta.scoring.scorer import score
from selecta.sequencing.arc import arc_target, load_profile

_WEIGHT_KEYS = ("harmonic", "tempo", "energy", "spectral", "phrase")
_DEFAULT_WEIGHTS = {key: 0.2 for key in _WEIGHT_KEYS}
_STRANDED_THRESHOLD = 25.0
_MAX_2OPT_PASSES = 50


@dataclass(frozen=True)
class OrderedTrack:
    track: TrackFeatures
    position: int
    arc_pos: float
    arc_target: float
    incoming_score: float | None


def sequence(
    tracks: list[TrackFeatures],
    profile: str = "experiential",
    weights: dict[str, float] | None = None,
    config_path: str | None = None,
) -> tuple[list[OrderedTrack], list[TrackFeatures]]:
    """Order tracks into a journey arc and flag weak incoming transitions."""

    if not tracks:
        return ([], [])

    points = load_profile(profile, config_path)
    active_weights = dict(_DEFAULT_WEIGHTS if weights is None else weights)

    # Rank-based arc mapping: real-world tracks (esp. mastered, uniformly-loud
    # genres like amapiano) have tightly-clustered absolute intensity, so matching
    # it directly against the arc gives almost no ordering signal. Replace each
    # track's intensity with its rank/percentile within THIS set, so the small but
    # real energy differences spread to fill the low->peak->resolve journey.
    tracks = _rank_intensities(tracks)

    ordered_tracks = _greedy_sequence(tracks, points, active_weights)
    ordered_tracks = _refine_with_2opt(ordered_tracks, points, active_weights)

    ordered = _build_ordered_tracks(ordered_tracks, points, active_weights)
    stranded = [
        item.track
        for item in ordered[1:]
        if item.incoming_score is not None and item.incoming_score < _STRANDED_THRESHOLD
    ]
    return (ordered, stranded)


def _rank_intensities(tracks: list[TrackFeatures]) -> list[TrackFeatures]:
    """Replace each track's intensity with its percentile rank within the set.

    The lowest-energy track gets 0.0, the highest 1.0, the rest spread evenly by
    rank. Other fields (bpm, key, title, phrases) are untouched. This makes the
    arc-adherence signal meaningful even when absolute intensities barely differ.
    """
    total = len(tracks)
    if total == 1:
        return [replace(tracks[0], intensity=0.5)]

    order = sorted(range(total), key=lambda i: tracks[i].intensity)
    percentile_by_index = {idx: rank / (total - 1) for rank, idx in enumerate(order)}
    return [
        replace(track, intensity=percentile_by_index[index])
        for index, track in enumerate(tracks)
    ]


def _greedy_sequence(
    tracks: list[TrackFeatures],
    points: list[tuple[float, float]],
    weights: dict[str, float],
) -> list[TrackFeatures]:
    total_tracks = len(tracks)
    start_target = arc_target(0.0, points)
    first_track = min(tracks, key=lambda track: abs(track.intensity - start_target))

    ordered = [first_track]
    unused = [track for track in tracks if track.path != first_track.path]
    current = first_track

    for position in range(1, total_tracks):
        slot_target = arc_target(_arc_pos(position, total_tracks), points)
        next_track = max(
            unused,
            key=lambda candidate: score(current, candidate, slot_target, weights).total,
        )
        ordered.append(next_track)
        unused = [track for track in unused if track.path != next_track.path]
        current = next_track

    return ordered


def _refine_with_2opt(
    tracks: list[TrackFeatures],
    points: list[tuple[float, float]],
    weights: dict[str, float],
) -> list[TrackFeatures]:
    best_order = list(tracks)
    best_score = _objective(best_order, points, weights)

    for _ in range(_MAX_2OPT_PASSES):
        improved = False
        for start in range(len(best_order) - 1):
            for end in range(start + 1, len(best_order)):
                candidate = (
                    best_order[:start]
                    + list(reversed(best_order[start : end + 1]))
                    + best_order[end + 1 :]
                )
                candidate_score = _objective(candidate, points, weights)
                if candidate_score > best_score:
                    best_order = candidate
                    best_score = candidate_score
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break

    return best_order


def _objective(
    tracks: list[TrackFeatures],
    points: list[tuple[float, float]],
    weights: dict[str, float],
) -> float:
    total_tracks = len(tracks)
    if total_tracks == 0:
        return 0.0

    adherence_sum = fsum(
        1.0 - abs(track.intensity - arc_target(_arc_pos(position, total_tracks), points))
        for position, track in enumerate(tracks)
    )
    transition_sum = fsum(
        score(
            tracks[position - 1],
            tracks[position],
            arc_target(_arc_pos(position, total_tracks), points),
            weights,
        ).total
        for position in range(1, total_tracks)
    )
    return transition_sum + (100.0 * adherence_sum / total_tracks)


def _build_ordered_tracks(
    tracks: list[TrackFeatures],
    points: list[tuple[float, float]],
    weights: dict[str, float],
) -> list[OrderedTrack]:
    total_tracks = len(tracks)
    ordered: list[OrderedTrack] = []

    for position, track in enumerate(tracks):
        pos = _arc_pos(position, total_tracks)
        target = arc_target(pos, points)
        incoming_score = None
        if position > 0:
            incoming_score = score(tracks[position - 1], track, target, weights).total
        ordered.append(
            OrderedTrack(
                track=track,
                position=position,
                arc_pos=pos,
                arc_target=target,
                incoming_score=incoming_score,
            )
        )

    return ordered


def _arc_pos(position: int, total_tracks: int) -> float:
    if total_tracks <= 1:
        return 0.0
    return position / (total_tracks - 1)
