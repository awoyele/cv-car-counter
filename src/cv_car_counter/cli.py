"""Command-line entry point for the CV Car Counter."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import load_camera_config
from .pipeline import run_pipeline


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="cv-car-counter",
        description="Count cars crossing configured gates in a 20-second traffic-video interval.",
    )
    p.add_argument("--video", required=True, help="path or URL to the source video")
    p.add_argument("--start", type=float, default=0.0, help="interval start in seconds")
    p.add_argument(
        "--duration",
        type=float,
        default=20.0,
        help="interval duration in seconds (default 20)",
    )
    p.add_argument("--config", required=True, help="camera config YAML path")
    p.add_argument(
        "--detector",
        default="yolo",
        choices=("yolo", "rfdtr"),
        help="detector backend (default yolo)",
    )
    p.add_argument("--weights", default=None, help="detector weights/checkpoint override")
    p.add_argument("--analysis-fps", type=int, default=15, help="CFR analysis frame rate")
    p.add_argument("--output", required=True, help="output directory")
    p.add_argument(
        "--no-annotated",
        action="store_true",
        help="skip writing the diagnostic annotated MP4",
    )
    return p


_RFDTR_SIZES = frozenset({"nano", "small", "medium", "large"})


def _detector_kwargs(detector: str, weights: str | None) -> dict:
    """Map ``--weights`` onto the selected backend without changing YOLO's contract.

    YOLO still receives ``weights=``. RF-DETR treats a size alias (nano/small/
    medium/large, with or without an ``rfdetr-`` prefix) as ``size=`` and any
    other value as a checkpoint path.
    """
    if not weights:
        return {}
    if detector == "yolo":
        return {"weights": weights}
    size_key = weights.lower().removeprefix("rfdetr-")
    if size_key in _RFDTR_SIZES:
        return {"size": size_key}
    return {"checkpoint": weights}


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_camera_config(args.config)

    detector_kwargs = _detector_kwargs(args.detector, args.weights)

    result = run_pipeline(
        video=args.video,
        start_seconds=args.start,
        duration_seconds=args.duration,
        config=config,
        detector_name=args.detector,
        analysis_fps=args.analysis_fps,
        output_dir=args.output,
        detector_kwargs=detector_kwargs,
        write_annotated=not args.no_annotated,
    )
    print(f"report:  {result.report_path}")
    print(f"clip:    {result.canonical_clip}")
    if result.annotated_clip:
        print(f"annotated: {result.annotated_clip}")
    print(f"car_count: {result.report['car_count']}")
    print(f"counts_by_direction: {result.report['counts_by_direction']}")
    if result.report["quality_flags"]:
        print(f"quality_flags: {result.report['quality_flags']}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
