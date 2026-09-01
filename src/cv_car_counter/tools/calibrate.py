"""Interactive calibration utility for drawing lane polygons and ENTRY/EXIT gates.

Usage:
    cv-car-calibrate --video videos/13361265-hd_1920_1080_60fps.mp4 \
        --config configs/pexels_13361265.yaml

The utility opens a reference frame and lets you click points to build lane
polygons and ordered gates, then writes the result back to the config YAML.
Run it after probing a new clip; never ship a config whose geometry was only
guessed from the plan.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml


def _pick_reference_frame(video: str, at_seconds: float):
    import av
    import numpy as np

    container = av.open(video)
    stream = container.streams.video[0]
    target = int(at_seconds * float(stream.average_rate or 30))
    frame_img = None
    for i, frame in enumerate(container.decode(stream)):
        if i >= target:
            frame_img = frame.to_ndarray(format="bgr24")
            break
    container.close()
    if frame_img is None:
        raise RuntimeError("could not extract a reference frame")
    return frame_img


def _click_points(window: str, image, n: int, label: str):
    import cv2
    import numpy as np

    points: list[tuple[int, int]] = []

    def on_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < n:
            points.append((x, y))
            disp = image.copy()
            for p in points:
                cv2.circle(disp, p, 5, (0, 255, 255), -1)
            if len(points) > 1:
                cv2.polylines(disp, [np.array(points)], False, (0, 255, 255), 2)
            cv2.putText(disp, f"{label}: click {n - len(points)} more",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow(window, disp)

    cv2.imshow(window, image)
    cv2.setMouseCallback(window, on_click)
    print(f"[calibrate] {label}: click {n} points, then press any key")
    while len(points) < n:
        if cv2.waitKey(50) & 0xFF == 27:  # ESC to cancel
            break
    cv2.destroyWindow(window)
    return [(float(x), float(y)) for x, y in points]


def calibrate(video: str, config_path: str, at_seconds: float = 1.0) -> None:
    import cv2

    path = Path(config_path)
    raw = yaml.safe_load(path.read_text()) if path.exists() else {"lanes": []}
    raw.setdefault("lanes", [])

    frame = _pick_reference_frame(video, at_seconds)
    cv2.namedWindow("calibrate", cv2.WINDOW_NORMAL)

    while True:
        lane_id = input("lane id (or blank to finish): ").strip()
        if not lane_id:
            break
        polygon = _click_points("calibrate", frame, n=4, label=f"{lane_id} polygon")
        if len(polygon) < 3:
            print("  polygon needs >=3 points, skipping lane")
            continue
        entry = _click_points("calibrate", frame, n=2, label=f"{lane_id} ENTRY gate")
        exit_ = _click_points("calibrate", frame, n=2, label=f"{lane_id} EXIT gate")
        direction = input("travel direction [dx dy] (e.g. '1 0' for L->R): ").strip()
        dx, dy = (direction.split() + ["0", "0"])[:2]
        raw["lanes"].append({
            "id": lane_id,
            "polygon": polygon,
            "entry_gate": entry,
            "exit_gate": exit_,
            "travel_direction": [float(dx), float(dy)],
        })

    raw.setdefault("camera_id", "pexels-13361265")
    raw.setdefault("version", "0")
    raw.setdefault("frame_size", [int(frame.shape[1]), int(frame.shape[0])])
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    print(f"[calibrate] wrote {path}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="cv-car-calibrate")
    p.add_argument("--video", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--at-seconds", type=float, default=1.0)
    args = p.parse_args(argv)
    calibrate(args.video, args.config, at_seconds=args.at_seconds)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
