from __future__ import annotations

import textwrap

import pytest

from cv_car_counter.config import (
    CameraConfig,
    Lane,
    TrackerParams,
    load_camera_config,
    parse_camera_config,
)


def _valid_lane_raw(lane_id="lane-1", direction=(1, 0)):
    return {
        "id": lane_id,
        "polygon": [[0, 0], [100, 0], [100, 100], [0, 100]],
        "entry_gate": [[10, 0], [10, 100]],
        "exit_gate": [[90, 0], [90, 100]],
        "travel_direction": list(direction),
    }


def test_parse_minimal_config():
    raw = {
        "camera_id": "cam-1",
        "version": "1",
        "frame_size": [1920, 1080],
        "lanes": [_valid_lane_raw()],
    }
    cfg = parse_camera_config(raw)
    assert isinstance(cfg, CameraConfig)
    assert cfg.camera_id == "cam-1"
    assert cfg.frame_size == (1920, 1080)
    assert len(cfg.lanes) == 1
    lane = cfg.lanes[0]
    assert isinstance(lane, Lane)
    assert lane.entry_gate == [(10.0, 0.0), (10.0, 100.0)]
    assert lane.travel_direction == (1.0, 0.0)


def test_travel_direction_is_normalized():
    raw = {"camera_id": "c", "frame_size": [10, 10], "lanes": [
        _valid_lane_raw(direction=(3, 0))
    ]}
    cfg = parse_camera_config(raw)
    assert cfg.lanes[0].travel_direction == (1.0, 0.0)


def test_missing_lanes_rejected():
    with pytest.raises(ValueError, match="at least one lane"):
        parse_camera_config({"camera_id": "c", "frame_size": [10, 10]})


def test_bad_gate_length_rejected():
    raw = {"camera_id": "c", "frame_size": [10, 10], "lanes": [
        {**_valid_lane_raw(), "entry_gate": [[0, 0]]},
    ]}
    with pytest.raises(ValueError, match="gate"):
        parse_camera_config(raw)


def test_zero_direction_rejected():
    raw = {"camera_id": "c", "frame_size": [10, 10], "lanes": [
        _valid_lane_raw(direction=(0, 0)),
    ]}
    with pytest.raises(ValueError, match="non-zero"):
        parse_camera_config(raw)


def test_tracker_defaults_and_override():
    raw = {"camera_id": "c", "frame_size": [10, 10], "lanes": [_valid_lane_raw()]}
    cfg = parse_camera_config(raw)
    assert isinstance(cfg.tracker, TrackerParams)
    assert cfg.tracker.line_dead_band_px == 6.0
    raw["tracker"] = {"line_dead_band_px": 12}
    cfg = parse_camera_config(raw)
    assert cfg.tracker.line_dead_band_px == 12.0


def test_load_from_yaml_file(tmp_path):
    p = tmp_path / "cam.yaml"
    p.write_text(textwrap.dedent("""
        camera_id: cam-yaml
        version: "2"
        frame_size: [640, 480]
        lanes:
          - id: lane-a
            polygon: [[0,0],[100,0],[100,100],[0,100]]
            entry_gate: [[10,0],[10,100]]
            exit_gate: [[90,0],[90,100]]
            travel_direction: [1, 0]
    """), encoding="utf-8")
    cfg = load_camera_config(p)
    assert cfg.camera_id == "cam-yaml"
    assert cfg.version == "2"
    assert cfg.lane_by_id("lane-a") is not None
    assert cfg.lane_by_id("nope") is None


def test_polygon_too_short_rejected():
    raw = {"camera_id": "c", "frame_size": [10, 10], "lanes": [
        {**_valid_lane_raw(), "polygon": [[0, 0], [10, 10]]},
    ]}
    with pytest.raises(ValueError, match="polygon"):
        parse_camera_config(raw)
