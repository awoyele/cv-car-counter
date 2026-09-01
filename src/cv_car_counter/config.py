"""Camera configuration schema and YAML loading.

A camera config describes one stationary scene: frame size, one or more lanes
(each with its own polygon, ordered ENTRY/EXIT gates, and travel direction),
ignore polygons, and tracker thresholds. Configs are versioned YAML files so a
report can always be tied back to the exact geometry that produced it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import yaml

Point = tuple[float, float]
Polygon = list[Point]
Segment = list[Point]  # a gate is a 2-point segment


@dataclass(frozen=True)
class Lane:
    id: str
    polygon: Polygon
    entry_gate: Segment
    exit_gate: Segment
    travel_direction: tuple[float, float]  # unit-ish vector in image coords

    def __post_init__(self) -> None:
        if len(self.entry_gate) != 2 or len(self.exit_gate) != 2:
            raise ValueError(f"lane {self.id!r}: gates must be 2-point segments")
        if len(self.polygon) < 3:
            raise ValueError(f"lane {self.id!r}: polygon needs >=3 points")


@dataclass(frozen=True)
class TrackerParams:
    line_dead_band_px: float = 6.0
    minimum_track_age_seconds: float = 0.3
    maximum_track_gap_seconds: float = 0.7
    minimum_directional_displacement_px: float = 8.0


@dataclass(frozen=True)
class CameraConfig:
    camera_id: str
    version: str
    frame_size: tuple[int, int]  # (width, height)
    lanes: list[Lane]
    ignore_polygons: list[Polygon] = field(default_factory=list)
    tracker: TrackerParams = field(default_factory=TrackerParams)

    def lane_by_id(self, lane_id: str) -> Lane | None:
        return next((l for l in self.lanes if l.id == lane_id), None)


def _as_point(seq: Sequence) -> Point:
    if len(seq) != 2:
        raise ValueError(f"expected [x, y] point, got {seq!r}")
    return (float(seq[0]), float(seq[1]))


def _as_polygon(raw: Sequence[Sequence]) -> Polygon:
    if not raw:
        return []
    return [_as_point(p) for p in raw]


def _as_segment(raw: Sequence[Sequence]) -> Segment:
    if len(raw) != 2:
        raise ValueError(f"gate must have exactly 2 points, got {raw!r}")
    return [_as_point(p) for p in raw]


def _as_direction(raw: Sequence) -> tuple[float, float]:
    if len(raw) != 2:
        raise ValueError(f"travel_direction must be [dx, dy], got {raw!r}")
    dx, dy = float(raw[0]), float(raw[1])
    mag = (dx * dx + dy * dy) ** 0.5
    if mag == 0:
        raise ValueError("travel_direction must be non-zero")
    return (dx / mag, dy / mag)


def load_camera_config(path: str | Path) -> CameraConfig:
    text = Path(path).read_text(encoding="utf-8")
    raw = yaml.safe_load(text) or {}
    return parse_camera_config(raw, source=str(path))


def parse_camera_config(raw: dict, source: str = "<inline>") -> CameraConfig:
    try:
        camera_id = raw["camera_id"]
        version = str(raw.get("version", "0"))
        width, height = raw["frame_size"]
        lanes_raw = raw.get("lanes") or []
        if not lanes_raw:
            raise ValueError("at least one lane is required")
        lanes = [
            Lane(
                id=l["id"],
                polygon=_as_polygon(l["polygon"]),
                entry_gate=_as_segment(l["entry_gate"]),
                exit_gate=_as_segment(l["exit_gate"]),
                travel_direction=_as_direction(l["travel_direction"]),
            )
            for l in lanes_raw
        ]
        tracker_raw = raw.get("tracker") or {}
        tracker = TrackerParams(
            line_dead_band_px=float(tracker_raw.get("line_dead_band_px", 6.0)),
            minimum_track_age_seconds=float(
                tracker_raw.get("minimum_track_age_seconds", 0.3)
            ),
            maximum_track_gap_seconds=float(
                tracker_raw.get("maximum_track_gap_seconds", 0.7)
            ),
            minimum_directional_displacement_px=float(
                tracker_raw.get("minimum_directional_displacement_px", 8.0)
            ),
        )
        return CameraConfig(
            camera_id=str(camera_id),
            version=version,
            frame_size=(int(width), int(height)),
            lanes=lanes,
            ignore_polygons=[_as_polygon(p) for p in raw.get("ignore_polygons") or []],
            tracker=tracker,
        )
    except KeyError as e:
        raise ValueError(f"camera config {source!r} missing required key: {e}") from e
