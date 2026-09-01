"""Per-lane ENTRY->EXIT counting state machine.

Implements the track state machine from PLAN.md:

    TENTATIVE -> CONFIRMED -> ENTRY_SIDE_SEEN -> ENTRY_GATE_CROSSED
              -> EXIT_GATE_CROSSED -> COUNTED

Rules enforced here:
* require multiple real detections before confirming (handled by the tracker);
* require observations outside the dead band on both sides of each gate;
* count only the configured ENTRY->EXIT order (NEAR->FAR-equivalent is rejected);
* never re-arm an already counted track;
* never count a crossing supported only by tracker prediction;
* preserve state across short occlusions and confirm on the next real detection;
* mark unresolved tracks UNKNOWN rather than forcing a direction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable

from .config import CameraConfig, Lane
from .geometry import (
    classify_step,
    lane_for_point,
    point_in_polygon,
)
from .tracking import TrackStep


class TrackState(str, Enum):
    TENTATIVE = "TENTATIVE"
    CONFIRMED = "CONFIRMED"
    ENTRY_SIDE_SEEN = "ENTRY_SIDE_SEEN"
    ENTRY_GATE_CROSSED = "ENTRY_GATE_CROSSED"
    EXIT_GATE_CROSSED = "EXIT_GATE_CROSSED"
    COUNTED = "COUNTED"
    UNKNOWN = "UNKNOWN"


@dataclass
class CountingEvent:
    track_id: int
    crossing_timestamp: float
    lane: str
    direction: str
    class_name: str
    confidence: float


@dataclass
class _PerLaneState:
    state: TrackState = TrackState.TENTATIVE
    last_contact: tuple[float, float] | None = None
    entry_crossed_at: float | None = None
    lane_id: str | None = None


@dataclass
class Counter:
    config: CameraConfig
    _tracks: dict[int, _PerLaneState] = field(default_factory=dict)
    events: list[CountingEvent] = field(default_factory=list)
    unknown_track_ids: list[int] = field(default_factory=list)

    def ingest(self, step: TrackStep) -> CountingEvent | None:
        """Process one confirmed track step; return a CountingEvent if counted."""
        if not step.is_real_detection:
            # Never advance the state machine on a prediction-only step.
            return None

        cfg = self.config
        st = self._tracks.setdefault(step.track_id, _PerLaneState())

        if st.state == TrackState.COUNTED:
            return None  # never re-arm

        # Resolve the lane. When lanes share a polygon (a single carriageway
        # with two directions that overlap in image-y), assign by MOTION
        # direction the first time we see real displacement, then keep it.
        # This avoids perspective-driven lane-switch resets.
        on_road = lane_for_point(step.contact, cfg.lanes) is not None
        if not on_road:
            # Outside the road: keep state but don't advance. Parked cars and
            # cross-traffic stay here and never get counted.
            st.last_contact = step.contact
            return None

        if st.lane_id is None and st.last_contact is not None:
            # First real displacement: pick the lane whose travel direction
            # matches the motion (and that contains the current point).
            dx = step.contact[0] - st.last_contact[0]
            dy = step.contact[1] - st.last_contact[1]
            mag = (dx * dx + dy * dy) ** 0.5
            if mag >= cfg.tracker.minimum_directional_displacement_px:
                best = None
                for lane in cfg.lanes:
                    if lane_for_point(step.contact, [lane]) is None:
                        continue
                    ux, uy = lane.travel_direction
                    if dx * ux + dy * uy > 0:
                        best = lane
                        break
                if best is not None:
                    st.lane_id = best.id
                else:
                    # On the road but moving against every configured direction:
                    # wrong-way. Mark UNKNOWN rather than forcing a count.
                    st.state = TrackState.UNKNOWN
                    self.unknown_track_ids.append(step.track_id)
                    st.last_contact = step.contact
                    return None

        lane = cfg.lane_by_id(st.lane_id) if st.lane_id else None
        if lane is None:
            # Direction not yet determined: record and wait for displacement.
            st.last_contact = step.contact
            return None

        if st.last_contact is None:
            st.last_contact = step.contact
            return None

        prev = st.last_contact
        curr = step.contact
        result = classify_step(
            prev, curr, lane, cfg.tracker.line_dead_band_px,
            cfg.tracker.minimum_directional_displacement_px,
        )

        event = None
        if st.state in (TrackState.TENTATIVE, TrackState.CONFIRMED):
            st.state = TrackState.CONFIRMED
            if result.crossed_entry and result.direction_ok:
                st.state = TrackState.ENTRY_GATE_CROSSED
                st.entry_crossed_at = step.timestamp
            elif (result.crossed_entry or result.crossed_exit) and not result.direction_ok:
                # Wrong-way traversal of either gate before entry is confirmed.
                st.state = TrackState.UNKNOWN
                self.unknown_track_ids.append(step.track_id)
        elif st.state == TrackState.ENTRY_GATE_CROSSED:
            if (
                result.crossed_exit
                and result.direction_ok
                and st.entry_crossed_at is not None
            ):
                st.state = TrackState.EXIT_GATE_CROSSED
                event = CountingEvent(
                    track_id=step.track_id,
                    crossing_timestamp=step.timestamp,
                    lane=lane.id,
                    direction=_direction_name(lane),
                    class_name="car",
                    confidence=step.confidence,
                )
                st.state = TrackState.COUNTED
            elif (result.crossed_entry or result.crossed_exit) and not result.direction_ok:
                # Reversed direction mid-traversal: do not force a count.
                st.state = TrackState.UNKNOWN
                self.unknown_track_ids.append(step.track_id)

        st.last_contact = curr
        if event is not None:
            self.events.append(event)
        return event

    def finalize(self) -> None:
        """Mark any track that entered but never exited as UNKNOWN."""
        for tid, st in self._tracks.items():
            if st.state == TrackState.ENTRY_GATE_CROSSED:
                self.unknown_track_ids.append(tid)
                st.state = TrackState.UNKNOWN


def _direction_name(lane: Lane) -> str:
    dx, dy = lane.travel_direction
    if abs(dx) >= abs(dy):
        return "left_to_right" if dx > 0 else "right_to_left"
    return "top_to_bottom" if dy > 0 else "bottom_to_top"
