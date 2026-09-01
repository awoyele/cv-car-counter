"""Geometry helpers: point-in-polygon, gate crossing, direction classification.

All geometry happens in image coordinates. The "road-contact point" of a
detection is approximated by the bottom-center of its bounding box, which is the
standard proxy for where a vehicle's tires meet the road.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .config import Lane, Polygon, Segment, Point


def bottom_center(box: tuple[float, float, float, float]) -> Point:
    """xyxy -> bottom-center road-contact point."""
    x1, y1, x2, y2 = box
    return ((x1 + x2) * 0.5, y2)


def point_in_polygon(point: Point, polygon: Polygon) -> bool:
    """Even-odd ray-cast point-in-polygon test."""
    if len(polygon) < 3:
        return False
    x, y = point
    inside = False
    n = len(polygon)
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > y) != (yj > y)) and (
            x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-12) + xi
        ):
            inside = not inside
        j = i
    return inside


def point_in_any_polygon(point: Point, polygons: Iterable[Polygon]) -> bool:
    return any(point_in_polygon(point, p) for p in polygons)


def _cross_sign(p: Point, a: Point, b: Point) -> float:
    """Sign of the oriented area (a->b x a->p). Positive => left of a->b."""
    return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])


def signed_distance_to_segment(point: Point, seg: Segment) -> float:
    """Perpendicular distance from `point` to the infinite line through `seg`.

    Sign follows the segment's orientation: positive on the left side of a->b,
    negative on the right. Used to detect gate crossings without dead-band jitter.
    """
    (ax, ay), (bx, by) = seg
    dx, dy = bx - ax, by - ay
    length = (dx * dx + dy * dy) ** 0.5
    if length == 0:
        return 0.0
    return _cross_sign(point, seg[0], seg[1]) / length


def crossed_gate(
    prev: Point,
    curr: Point,
    gate: Segment,
    dead_band_px: float,
) -> bool:
    """True when the contact point moved from one side of `gate` to the other,
    with both observations outside the `dead_band_px` band around the gate."""
    d_prev = signed_distance_to_segment(prev, gate)
    d_curr = signed_distance_to_segment(curr, gate)
    if abs(d_prev) <= dead_band_px or abs(d_curr) <= dead_band_px:
        return False
    return d_prev * d_curr < 0


def displacement_vector(prev: Point, curr: Point) -> tuple[float, float]:
    return (curr[0] - prev[0], curr[1] - prev[1])


def matches_direction(
    prev: Point,
    curr: Point,
    travel_direction: tuple[float, float],
    minimum_displacement_px: float,
) -> bool:
    """True when motion from prev->curr aligns with the lane's travel direction
    by at least `minimum_displacement_px` of projected displacement."""
    dx, dy = displacement_vector(prev, curr)
    mag = (dx * dx + dy * dy) ** 0.5
    if mag < minimum_displacement_px:
        return False
    ux, uy = travel_direction
    projected = dx * ux + dy * uy
    return projected >= minimum_displacement_px


@dataclass(frozen=True)
class GateCrossing:
    """Result of checking one track step against a lane's ENTRY/EXIT gates."""

    crossed_entry: bool = False
    crossed_exit: bool = False
    inside_lane: bool = False
    direction_ok: bool = False


def classify_step(
    prev: Point,
    curr: Point,
    lane: Lane,
    dead_band_px: float,
    minimum_displacement_px: float,
) -> GateCrossing:
    return GateCrossing(
        crossed_entry=crossed_gate(prev, curr, lane.entry_gate, dead_band_px),
        crossed_exit=crossed_gate(prev, curr, lane.exit_gate, dead_band_px),
        inside_lane=point_in_polygon(curr, lane.polygon),
        direction_ok=matches_direction(
            prev, curr, lane.travel_direction, minimum_displacement_px
        ),
    )


def lane_for_point(point: Point, lanes: Iterable[Lane]) -> Lane | None:
    """First lane whose polygon contains the point (lanes should not overlap)."""
    for lane in lanes:
        if point_in_polygon(point, lane.polygon):
            return lane
    return None
