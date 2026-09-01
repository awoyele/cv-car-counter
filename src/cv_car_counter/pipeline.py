"""End-to-end pipeline: probe -> canonical clip -> detect -> track -> count -> report."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .annotation import write_annotated_video
from .config import CameraConfig
from .counting import Counter
from .detectors import Detector, get_detector
from .media import make_canonical_clip, probe, sha256_of_file, validate_canonical_clip
from .reporting import build_report, write_report
from .tracking import ByteTrackWrapper


@dataclass
class PipelineResult:
    report: dict
    canonical_clip: Path
    annotated_clip: Path | None
    report_path: Path


def _tracker_config_hash(config: CameraConfig) -> str:
    raw = (
        f"{config.tracker.line_dead_band_px},"
        f"{config.tracker.minimum_track_age_seconds},"
        f"{config.tracker.maximum_track_gap_seconds},"
        f"{config.tracker.minimum_directional_displacement_px}"
    ).encode()
    return hashlib.sha256(raw).hexdigest()[:16]


def run_pipeline(
    *,
    video: str | Path,
    start_seconds: float,
    duration_seconds: float,
    config: CameraConfig,
    detector_name: str,
    analysis_fps: int = 15,
    output_dir: str | Path,
    detector_kwargs: dict | None = None,
    write_annotated: bool = True,
) -> PipelineResult:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    probe_result = probe(video)
    if probe_result.duration_seconds < start_seconds + duration_seconds:
        raise ValueError(
            f"source ({probe_result.duration_seconds:.2f}s) is shorter than "
            f"requested interval end ({start_seconds + duration_seconds:.2f}s)"
        )

    canonical = output_dir / "clip.canonical.mp4"
    make_canonical_clip(video, start_seconds, duration_seconds, canonical, analysis_fps)
    clip_sha = sha256_of_file(canonical)

    detector: Detector = get_detector(detector_name, **(detector_kwargs or {}))
    tracker = ByteTrackWrapper(
        frame_rate=analysis_fps,
        minimum_track_age_seconds=config.tracker.minimum_track_age_seconds,
    )
    counter = Counter(config=config)

    frames: list[np.ndarray] = []
    steps_per_frame: list[list] = []
    frames_decoded = _decode_and_process(
        canonical, analysis_fps, duration_seconds, detector, tracker, counter,
        collect_frames=write_annotated, frames_out=frames, steps_out=steps_per_frame,
    )

    counter.finalize()

    report = build_report(
        clip_path=canonical,
        clip_sha256=clip_sha,
        start_seconds=start_seconds,
        duration_seconds=duration_seconds,
        fps=analysis_fps,
        frames=frames_decoded,
        camera_config_version=config.version,
        model_version=detector.model_version,
        tracker_config_hash=_tracker_config_hash(config),
        counter=counter,
    )
    report_path = write_report(report, output_dir / "report.json")

    annotated = None
    if write_annotated and frames:
        annotated = output_dir / "clip.annotated.mp4"
        write_annotated_video(frames, config, steps_per_frame, counter, annotated, analysis_fps)

    return PipelineResult(
        report=report,
        canonical_clip=canonical,
        annotated_clip=annotated,
        report_path=report_path,
    )


def _decode_and_process(
    clip_path: Path,
    analysis_fps: int,
    duration_seconds: float,
    detector: Detector,
    tracker: ByteTrackWrapper,
    counter: Counter,
    *,
    collect_frames: bool,
    frames_out: list,
    steps_out: list,
) -> int:
    """Decode frames via PyAV (timestamp-aware) and run detect/track/count."""
    try:
        import av
    except ImportError as e:  # pragma: no cover
        raise ImportError("PyAV (av) is required for decoding. pip install -e .") from e

    container = av.open(str(clip_path))
    stream = container.streams.video[0]
    expected = int(round(duration_seconds * analysis_fps))
    count = 0
    for frame in container.decode(stream):
        img = frame.to_ndarray(format="bgr24")
        ts = count / analysis_fps  # PTS reset to 0 in the canonical clip
        detections = detector.detect(img)
        steps = tracker.update(detections, img, ts)
        for step in steps:
            counter.ingest(step)
        if collect_frames:
            frames_out.append(img)
            steps_out.append(steps)
        count += 1
        if count >= expected:
            break
    container.close()
    return count
