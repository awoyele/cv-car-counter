"""ByteTrack wrapper that turns raw detections into persistent tracks.

We use Roboflow's ``supervision`` ByteTrack implementation. Each track keeps a
stable id and the most recent road-contact point, which the counting state
machine consumes. Tracker prediction-only updates are flagged so the counter
never counts a crossing that was not supported by a real detection.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .detectors import Detection
from .geometry import bottom_center


@dataclass
class TrackStep:
    """One observation of one track at one timestamp."""

    track_id: int
    timestamp: float
    box: tuple[float, float, float, float]
    contact: tuple[float, float]
    confidence: float
    is_real_detection: bool  # False when the position came from tracker prediction


@dataclass
class TrackHistory:
    track_id: int
    steps: list[TrackStep] = field(default_factory=list)

    def last_real_step(self) -> TrackStep | None:
        for s in reversed(self.steps):
            if s.is_real_detection:
                return s
        return None


class ByteTrackWrapper:
    """Thin adapter around ``supervision.ByteTrack``."""

    def __init__(
        self,
        frame_rate: float,
        minimum_consecutive_frames: int = 3,
        minimum_track_age_seconds: float = 0.3,
    ) -> None:
        try:
            from supervision import ByteTrack
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "supervision is required for tracking. pip install -e ."
            ) from e
        self._tracker = ByteTrack(frame_rate=int(round(frame_rate)))
        self._minimum_consecutive_frames = minimum_consecutive_frames
        self._minimum_track_age_seconds = minimum_track_age_seconds
        self._raw_steps: dict[int, list[TrackStep]] = {}
        self._confirmed: set[int] = set()

    def update(
        self, detections: list[Detection], frame: np.ndarray, timestamp: float
    ) -> list[TrackStep]:
        from supervision import Detections as SVDetections

        if detections:
            xyxy = np.array([d.box for d in detections], dtype=float)
            conf = np.array([d.confidence for d in detections], dtype=float)
            cls = np.array([d.class_id for d in detections], dtype=int)
            sv = SVDetections(xyxy=xyxy, confidence=conf, class_id=cls)
        else:
            sv = SVDetections.empty()

        tracked = self._tracker.update_with_detections(sv)
        steps: list[TrackStep] = []
        if len(tracked) == 0:
            return steps

        tids = tracked.tracker_id.astype(int)
        xyxy = tracked.xyxy
        conf = tracked.confidence
        # supervision does not flag predicted vs. detected; treat any returned
        # box as a real observation for this frame. The counter still requires
        # multiple real detections before confirming a track.
        for box, tid, c in zip(xyxy, tids, conf):
            tid = int(tid)
            box_t = (float(box[0]), float(box[1]), float(box[2]), float(box[3]))
            step = TrackStep(
                track_id=tid,
                timestamp=timestamp,
                box=box_t,
                contact=bottom_center(box_t),
                confidence=float(c),
                is_real_detection=True,
            )
            self._raw_steps.setdefault(tid, []).append(step)
            # Emit every real-detection step so the counter can evaluate gate
            # crossings across the full trajectory. The confirmation gate used
            # to live here, but it caused fast cars to cross the entry gate before
            # the counter ever saw them. The counter's ENTRY->EXIT requirement
            # (two ordered crossings with direction checks) already filters
            # spurious 1-frame tracks, so emitting early is safe.
            if self._is_confirmed(tid, timestamp):
                self._confirmed.add(tid)
            steps.append(step)
        return steps

    def _is_confirmed(self, tid: int, timestamp: float) -> bool:
        history = self._raw_steps[tid]
        if len(history) < self._minimum_consecutive_frames:
            return False
        age = timestamp - history[0].timestamp
        return age >= self._minimum_track_age_seconds

    @property
    def confirmed_ids(self) -> set[int]:
        return set(self._confirmed)

    def history_for(self, tid: int) -> TrackHistory:
        return TrackHistory(tid, list(self._raw_steps.get(tid, [])))
