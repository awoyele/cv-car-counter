"""JSON report writer.

The report is the authoritative output. It carries the clip hash, source
interval, timing, config/model versions, total count, per-direction totals,
and one entry per crossing event. Quality flags surface anything that should
block trust in the count (e.g. unresolved UNKNOWN tracks).
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .counting import Counter, CountingEvent


def build_report(
    *,
    clip_path: str | Path,
    clip_sha256: str,
    start_seconds: float,
    duration_seconds: float,
    fps: int,
    frames: int,
    camera_config_version: str,
    model_version: str,
    tracker_config_hash: str,
    counter: Counter,
) -> dict[str, Any]:
    counts_by_direction: dict[str, int] = {}
    for ev in counter.events:
        counts_by_direction[ev.direction] = counts_by_direction.get(ev.direction, 0) + 1

    quality_flags: list[str] = []
    if counter.unknown_track_ids:
        quality_flags.append(
            f"unresolved_tracks:{len(counter.unknown_track_ids)}"
        )

    return {
        "clip_id": f"sha256:{clip_sha256}",
        "clip_path": str(clip_path),
        "source_interval": {
            "start_seconds": round(start_seconds, 6),
            "duration_seconds": round(duration_seconds, 6),
        },
        "timing": {
            "fps": fps,
            "frames": frames,
            "duration_seconds": round(duration_seconds, 6),
        },
        "camera_config_version": camera_config_version,
        "model_version": model_version,
        "tracker_config_hash": tracker_config_hash,
        "car_count": len(counter.events),
        "counts_by_direction": counts_by_direction,
        "events": [asdict(ev) for ev in counter.events],
        "unknown_track_ids": list(counter.unknown_track_ids),
        "quality_flags": quality_flags,
    }


def write_report(report: dict[str, Any], out_path: str | Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return out_path
