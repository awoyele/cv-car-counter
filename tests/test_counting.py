from __future__ import annotations

import pytest

from cv_car_counter.config import CameraConfig, Lane, TrackerParams
from cv_car_counter.counting import Counter, TrackState
from cv_car_counter.tracking import TrackStep


def _lane(direction=(1, 0)) -> Lane:
    # Place gates according to travel direction: ENTRY is the gate a car
    # reaches first, EXIT is the one it reaches last.
    if direction == (1, 0):  # left -> right
        entry_gate = [(10.0, 0.0), (10.0, 100.0)]
        exit_gate = [(90.0, 0.0), (90.0, 100.0)]
    else:  # right -> left
        entry_gate = [(90.0, 0.0), (90.0, 100.0)]
        exit_gate = [(10.0, 0.0), (10.0, 100.0)]
    return Lane(
        id="lane-1",
        polygon=[(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)],
        entry_gate=entry_gate,
        exit_gate=exit_gate,
        travel_direction=direction,
    )


def _config(direction=(1, 0)) -> CameraConfig:
    return CameraConfig(
        camera_id="test",
        version="0",
        frame_size=(100, 100),
        lanes=[_lane(direction)],
        tracker=TrackerParams(
            line_dead_band_px=2,
            minimum_directional_displacement_px=8,
        ),
    )


def _step(tid, t, x, y, conf=0.9, real=True) -> TrackStep:
    return TrackStep(
        track_id=tid,
        timestamp=t,
        box=(x - 5, y - 5, x + 5, y + 5),
        contact=(x, y),
        confidence=conf,
        is_real_detection=real,
    )


def test_full_entry_to_exit_counts_once():
    cfg = _config(direction=(1, 0))
    counter = Counter(config=cfg)
    # approach entry from outside
    counter.ingest(_step(1, 0.0, 5, 50))
    counter.ingest(_step(1, 0.1, 50, 50))   # crosses entry going +x
    ev = counter.ingest(_step(1, 0.2, 95, 50))  # crosses exit going +x
    assert ev is not None
    assert ev.track_id == 1
    assert ev.direction == "left_to_right"
    assert counter.events == [ev]
    # subsequent steps do not re-count
    assert counter.ingest(_step(1, 0.3, 99, 50)) is None
    assert len(counter.events) == 1


def test_wrong_way_not_counted_and_marked_unknown():
    cfg = _config(direction=(1, 0))  # only L->R allowed
    counter = Counter(config=cfg)
    counter.ingest(_step(1, 0.0, 95, 50))   # start on the EXIT side
    # cross the exit gate going -x (wrong way): direction_ok False
    ev = counter.ingest(_step(1, 0.1, 50, 50))
    assert ev is None
    assert counter._tracks[1].state == TrackState.UNKNOWN
    assert 1 in counter.unknown_track_ids


def test_prediction_only_step_does_not_advance():
    cfg = _config(direction=(1, 0))
    counter = Counter(config=cfg)
    counter.ingest(_step(1, 0.0, 5, 50))
    pred = _step(1, 0.1, 50, 50, real=False)
    assert counter.ingest(pred) is None
    # state should still be TENTATIVE/CONFIRMED, not ENTRY_GATE_CROSSED
    assert counter._tracks[1].state != TrackState.ENTRY_GATE_CROSSED


def test_outside_lane_does_not_count():
    cfg = _config(direction=(1, 0))
    counter = Counter(config=cfg)
    counter.ingest(_step(1, 0.0, 500, 500))  # outside polygon
    counter.ingest(_step(1, 0.1, 600, 500))
    assert counter.events == []


def test_finalize_marks_unfinished_as_unknown():
    cfg = _config(direction=(1, 0))
    counter = Counter(config=cfg)
    counter.ingest(_step(1, 0.0, 5, 50))
    counter.ingest(_step(1, 0.1, 50, 50))  # crossed entry, no exit
    counter.finalize()
    assert counter._tracks[1].state == TrackState.UNKNOWN
    assert 1 in counter.unknown_track_ids


def test_right_to_left_direction_label():
    cfg = _config(direction=(-1, 0))  # entry at x=90, exit at x=10
    counter = Counter(config=cfg)
    counter.ingest(_step(1, 0.0, 95, 50))   # approach from the right (entry side)
    counter.ingest(_step(1, 0.1, 50, 50))   # crosses ENTRY gate (x=90) going -x
    ev = counter.ingest(_step(1, 0.2, 5, 50))   # crosses EXIT gate (x=10) going -x
    assert ev is not None
    assert ev.direction == "right_to_left"
