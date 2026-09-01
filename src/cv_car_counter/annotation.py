"""Annotated MP4 writer for diagnostic output.

The annotated video is a debugging artifact, not the source of truth. It draws
lane polygons, ENTRY/EXIT gates, detection boxes, track ids, and the live state
of each track so a human can audit why a car was or was not counted.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .config import CameraConfig, Lane
from .counting import Counter, TrackState
from .tracking import TrackStep


_LANE_COLOR = (255, 255, 0)      # yellow
_ENTRY_COLOR = (0, 255, 0)       # green
_EXIT_COLOR = (0, 0, 255)        # red
_BOX_COLOR = (80, 200, 255)
_COUNTED_COLOR = (0, 255, 120)
_UNKNOWN_COLOR = (0, 0, 200)


def _draw_polygon(img, polygon, color, thickness=2):
    import cv2

    if len(polygon) < 2:
        return
    pts = np.array(polygon, dtype=np.int32).reshape(-1, 1, 2)
    cv2.polylines(img, [pts], isClosed=True, color=color, thickness=thickness)


def _draw_segment(img, seg, color, thickness=2):
    import cv2

    (x1, y1), (x2, y2) = seg
    cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)


def draw_overlay(
    frame: np.ndarray,
    config: CameraConfig,
    steps: list[TrackStep],
    counter: Counter,
) -> np.ndarray:
    import cv2

    out = frame.copy()
    for lane in config.lanes:
        _draw_polygon(out, lane.polygon, _LANE_COLOR, thickness=2)
        _draw_segment(out, lane.entry_gate, _ENTRY_COLOR, thickness=3)
        _draw_segment(out, lane.exit_gate, _EXIT_COLOR, thickness=3)
        # label the lane near its first polygon vertex
        if lane.polygon:
            x, y = lane.polygon[0]
            cv2.putText(
                out, lane.id, (int(x), int(y) - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, _LANE_COLOR, 2,
            )
    for p in config.ignore_polygons:
        _draw_polygon(out, p, (90, 90, 90), thickness=1)

    for s in steps:
        x1, y1, x2, y2 = s.box
        st = counter._tracks.get(s.track_id)
        state = st.state if st else TrackState.TENTATIVE
        color = _COUNTED_COLOR if state == TrackState.COUNTED else (
            _UNKNOWN_COLOR if state == TrackState.UNKNOWN else _BOX_COLOR
        )
        cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.putText(
            out, f"#{s.track_id}", (int(x1), max(0, int(y1) - 6)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2,
        )
    return out


def write_annotated_video(
    frames: list[np.ndarray],
    config: CameraConfig,
    steps_per_frame: list[list[TrackStep]],
    counter: Counter,
    out_path: str | Path,
    fps: int,
) -> Path:
    """Render an annotated MP4 from in-memory frames. Diagnostic use only.

    Encodes with H.264 (libx264) via PyAV rather than cv2.VideoWriter's
    ``mp4v`` codec: MPEG-4 Part 2 inside an MP4 container is rejected by most
    modern players (macOS QuickTime, browsers, iOS), which expect H.264.
    """
    import av

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    w, h = config.frame_size[0], config.frame_size[1]

    container = av.open(str(out_path), mode="w", format="mp4")
    stream = container.add_stream("libx264", rate=fps)
    stream.width = w
    stream.height = h
    stream.pix_fmt = "yuv420p"
    stream.options = {"preset": "fast", "crf": "20"}

    for frame, steps in zip(frames, steps_per_frame):
        annotated = draw_overlay(frame, config, steps, counter)
        # cv2 gives BGR; PyAV wants RGB for the frame helper.
        arr = annotated[:, :, ::-1]
        av_frame = av.VideoFrame.from_ndarray(arr, format="rgb24")
        for packet in stream.encode(av_frame):
            container.mux(packet)
    # Flush the encoder.
    for packet in stream.encode():
        container.mux(packet)
    container.close()
    return out_path
