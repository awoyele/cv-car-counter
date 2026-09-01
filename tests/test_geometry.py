from __future__ import annotations

import pytest

from cv_car_counter.config import Lane
from cv_car_counter.geometry import (
    bottom_center,
    classify_step,
    crossed_gate,
    lane_for_point,
    matches_direction,
    point_in_any_polygon,
    point_in_polygon,
    signed_distance_to_segment,
)


def _lane(direction=(1, 0)) -> Lane:
    return Lane(
        id="lane-1",
        polygon=[(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)],
        entry_gate=[(10.0, 0.0), (10.0, 100.0)],
        exit_gate=[(90.0, 0.0), (90.0, 100.0)],
        travel_direction=direction,
    )


def test_bottom_center():
    assert bottom_center((10, 20, 30, 40)) == (20.0, 40.0)


def test_point_in_polygon_inside_and_outside():
    poly = [(0, 0), (100, 0), (100, 100), (0, 100)]
    assert point_in_polygon((50, 50), poly)
    assert not point_in_polygon((150, 50), poly)
    assert not point_in_polygon((50, -1), poly)


def test_point_in_polygon_degenerate():
    assert not point_in_polygon((0, 0), [(0, 0), (1, 1)])


def test_point_in_any_polygon():
    a = [(0, 0), (10, 0), (10, 10), (0, 10)]
    b = [(20, 20), (30, 20), (30, 30), (20, 30)]
    assert point_in_any_polygon((5, 5), [a, b])
    assert point_in_any_polygon((25, 25), [a, b])
    assert not point_in_any_polygon((15, 15), [a, b])


def test_signed_distance_to_segment_left_right():
    seg = [(0.0, 0.0), (10.0, 0.0)]  # horizontal, pointing +x
    assert signed_distance_to_segment((5, 5), seg) > 0   # above = left of +x
    assert signed_distance_to_segment((5, -5), seg) < 0  # below = right of +x


def test_crossed_gate_detects_real_crossing():
    gate = [(50.0, 0.0), (50.0, 100.0)]  # vertical line at x=50
    assert crossed_gate((40, 50), (60, 50), gate, dead_band_px=2)
    # same side -> no crossing
    assert not crossed_gate((40, 50), (45, 50), gate, dead_band_px=2)
    # within dead band -> ignored
    assert not crossed_gate((49, 50), (51, 50), gate, dead_band_px=5)


def test_matches_direction_lateral():
    assert matches_direction((0, 5), (20, 5), (1, 0), minimum_displacement_px=8)
    assert not matches_direction((20, 5), (0, 5), (1, 0), minimum_displacement_px=8)
    # too small displacement
    assert not matches_direction((0, 5), (3, 5), (1, 0), minimum_displacement_px=8)


def test_classify_step_full_traversal():
    lane = _lane()
    # step from left of entry to between gates, going +x
    res = classify_step((5, 50), (50, 50), lane, dead_band_px=2, minimum_displacement_px=8)
    assert res.crossed_entry
    assert not res.crossed_exit
    assert res.inside_lane
    assert res.direction_ok
    # step from between gates to right of exit
    res2 = classify_step((50, 50), (95, 50), lane, dead_band_px=2, minimum_displacement_px=8)
    assert not res2.crossed_entry
    assert res2.crossed_exit
    assert res2.direction_ok


def test_lane_for_point_returns_first_match():
    l1 = _lane()
    l2 = Lane(
        id="lane-2",
        polygon=[(200.0, 0.0), (300.0, 0.0), (300.0, 100.0), (200.0, 100.0)],
        entry_gate=[(210.0, 0.0), (210.0, 100.0)],
        exit_gate=[(290.0, 0.0), (290.0, 100.0)],
        travel_direction=(1, 0),
    )
    assert lane_for_point((50, 50), [l1, l2]).id == "lane-1"
    assert lane_for_point((250, 50), [l1, l2]).id == "lane-2"
    assert lane_for_point((500, 50), [l1, l2]) is None
