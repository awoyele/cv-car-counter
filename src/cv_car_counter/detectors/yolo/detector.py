"""Ultralytics YOLO detector implementation.

This path is the fastest to prototype. By default we keep only the COCO
``car`` class (id 2). Pass ``classes`` to widen the set, e.g. for ablation
studies that compare car vs. motorcycle/truck rejection.
"""

from __future__ import annotations

from typing import Iterable

from ..base import Detection

# COCO class ids we accept as "car" for the counter. We deliberately exclude
# motorbike (3), bus (5), and truck (7) — see PLAN.md counting definition.
DEFAULT_CLASSES: tuple[int, ...] = (2,)  # car


class YoloDetector:
    name = "yolo"

    def __init__(
        self,
        weights: str = "yolov8n.pt",
        confidence: float = 0.3,
        classes: Iterable[int] = DEFAULT_CLASSES,
        imgsz: int | None = None,
    ) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as e:  # pragma: no cover - import-guard
            raise ImportError(
                "ultralytics is required for the yolo detector path. "
                "Install with: pip install -e ."
            ) from e
        self._model = YOLO(weights)
        self._weights = weights
        self._confidence = confidence
        self._classes = tuple(classes)
        self._imgsz = imgsz
        self.model_version = f"ultralytics:{weights}"

    def detect(self, frame) -> list[Detection]:
        kwargs = {"verbose": False, "conf": self._confidence}
        if self._imgsz is not None:
            kwargs["imgsz"] = self._imgsz
        results = self._model.predict(frame, **kwargs)
        out: list[Detection] = []
        for r in results:
            boxes = r.boxes
            if boxes is None or len(boxes) == 0:
                continue
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            cls_ids = boxes.cls.cpu().numpy().astype(int)
            for box, conf, cid in zip(xyxy, confs, cls_ids):
                if self._classes and int(cid) not in self._classes:
                    continue
                out.append(
                    Detection(
                        box=(float(box[0]), float(box[1]), float(box[2]), float(box[3])),
                        confidence=float(conf),
                        class_id=int(cid),
                    )
                )
        return out
