"""Journey arc profile loading and interpolation."""

from __future__ import annotations

from bisect import bisect_left
from pathlib import Path
from typing import Any
import tomllib


def load_profile(profile: str, config_path: str | None = None) -> list[tuple[float, float]]:
    """Load an arc profile from config as sorted ``(position, intensity)`` points."""
    path = Path(config_path) if config_path is not None else Path(__file__).resolve().parents[3] / "config.toml"

    with path.open("rb") as handle:
        config = tomllib.load(handle)

    arc_table = config.get("arc")
    if not isinstance(arc_table, dict):
        raise ValueError("Missing [arc] table in config.")

    profile_table = arc_table.get(profile)
    if profile_table is None:
        raise KeyError(profile)
    if not isinstance(profile_table, dict):
        raise ValueError(f"Arc profile {profile!r} must be a table.")

    raw_points = profile_table.get("points")
    if not isinstance(raw_points, list) or not raw_points:
        raise ValueError(f"Arc profile {profile!r} must define a non-empty points list.")

    points = [_coerce_point(point, profile=profile) for point in raw_points]
    return sorted(points, key=lambda point: point[0])


def arc_target(pos: float, points: list[tuple[float, float]]) -> float:
    """Return the interpolated target intensity for a normalised set position."""
    if not points:
        raise ValueError("points must not be empty")

    clamped_pos = min(max(float(pos), 0.0), 1.0)
    sorted_points = sorted(points, key=lambda point: point[0])

    if clamped_pos <= sorted_points[0][0]:
        return _clamp_unit(sorted_points[0][1])
    if clamped_pos >= sorted_points[-1][0]:
        return _clamp_unit(sorted_points[-1][1])

    positions = [point[0] for point in sorted_points]
    right_index = bisect_left(positions, clamped_pos)
    left_pos, left_intensity = sorted_points[right_index - 1]
    right_pos, right_intensity = sorted_points[right_index]

    if right_pos == left_pos:
        return _clamp_unit(right_intensity)

    span = right_pos - left_pos
    weight = (clamped_pos - left_pos) / span
    intensity = left_intensity + weight * (right_intensity - left_intensity)
    return _clamp_unit(intensity)


def _coerce_point(point: Any, *, profile: str) -> tuple[float, float]:
    if not isinstance(point, (list, tuple)) or len(point) != 2:
        raise ValueError(f"Arc profile {profile!r} contains an invalid point: {point!r}")

    try:
        pos = float(point[0])
        intensity = float(point[1])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Arc profile {profile!r} contains non-numeric point values: {point!r}") from exc

    return (_clamp_unit(pos), _clamp_unit(intensity))


def _clamp_unit(value: float) -> float:
    return min(max(value, 0.0), 1.0)
