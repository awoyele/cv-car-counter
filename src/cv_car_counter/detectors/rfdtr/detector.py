"""RF-DETR detector implementation.

RF-DETR offers permissive checkpoint terms and is the recommended
proprietary-friendly alternative to Ultralytics YOLO. The optional
``rfdtr`` extra installs the ``rfdetr`` package:

    pip install -e ".[rfdtr]"

``predict()`` returns a ``supervision.Detections`` object with xyxy boxes.
Input frames are BGR (same as the rest of this pipeline); RF-DETR expects
RGB, so we convert at the boundary.

Class ids follow original COCO category ids (``car`` is 3), not the
0-indexed YOLO mapping (``car`` is 2). We keep only cars by default.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np

from ..base import Detection

DEFAULT_CLASSES: tuple[int, ...] = (3,)  # original COCO category id for car

_SIZE_TO_CLASS = {
    "nano": "RFDETRNano",
    "small": "RFDETRSmall",
    "medium": "RFDETRMedium",
    "large": "RFDETRLarge",
}


def _default_device() -> str:
    """Prefer CUDA; skip MPS.

    RF-DETR 1.5.x on Apple's MPS backend currently yields near-zero scores
    (upsample ops fall back to CPU and the mixed-device graph is wrong). CPU
    is slower but returns real COCO detections.
    """
    try:
        import torch
    except ImportError:  # pragma: no cover
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _normalize_size(size: str) -> str:
    key = size.lower().removeprefix("rfdetr-")
    if key not in _SIZE_TO_CLASS:
        raise ValueError(
            f"unknown rfdetr size {size!r}; choose one of {sorted(_SIZE_TO_CLASS)}"
        )
    return key


class RfdetrDetector:
    name = "rfdtr"

    def __init__(
        self,
        checkpoint: str | None = None,
        confidence: float = 0.3,
        classes: Iterable[int] = DEFAULT_CLASSES,
        size: str = "nano",
        device: str | None = None,
    ) -> None:
        try:
            import rfdetr
        except ImportError as e:  # pragma: no cover - import-guard
            raise ImportError(
                "rfdetr is required for the rfdtr detector path. "
                "Install with: pip install -e \".[rfdtr]\""
            ) from e

        size_key = _normalize_size(size)
        cls_name = _SIZE_TO_CLASS[size_key]
        model_cls = getattr(rfdetr, cls_name, None) or getattr(rfdetr, "RFDETRBase", None)
        if model_cls is None:
            raise ImportError(
                f"rfdetr has no {cls_name} (or RFDETRBase fallback). "
                "Upgrade with: pip install -U rfdetr"
            )

        kwargs: dict = {}
        if checkpoint:
            kwargs["pretrain_weights"] = checkpoint
        kwargs["device"] = device or _default_device()
        self._model = model_cls(**kwargs)
        self._checkpoint = checkpoint or f"rfdetr-{size_key}"
        self._confidence = confidence
        self._classes = tuple(classes)
        self.model_version = f"rfdetr:{self._checkpoint}"

    def detect(self, frame) -> list[Detection]:
        rgb = np.ascontiguousarray(frame[:, :, ::-1])
        preds = self._model.predict(rgb, threshold=self._confidence)
        xyxy, confs, cls_ids = _as_xyxy_arrays(preds)
        out: list[Detection] = []
        for box, conf, cid in zip(xyxy, confs, cls_ids):
            class_id = int(cid)
            if self._classes and class_id not in self._classes:
                continue
            out.append(
                Detection(
                    box=(float(box[0]), float(box[1]), float(box[2]), float(box[3])),
                    confidence=float(conf),
                    class_id=class_id,
                )
            )
        return out


def _as_xyxy_arrays(preds) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Normalize ``predict()`` output to parallel xyxy / conf / class_id arrays."""
    if preds is None:
        return (
            np.empty((0, 4), dtype=float),
            np.empty((0,), dtype=float),
            np.empty((0,), dtype=int),
        )
    if hasattr(preds, "xyxy"):
        xyxy = np.asarray(preds.xyxy, dtype=float)
        n = 0 if xyxy.size == 0 else len(xyxy)
        confs = preds.confidence
        cls_ids = preds.class_id
        if confs is None:
            confs = np.ones(n, dtype=float)
        else:
            confs = np.asarray(confs, dtype=float)
        if cls_ids is None:
            cls_ids = np.full(n, -1, dtype=int)
        else:
            cls_ids = np.asarray(cls_ids, dtype=int)
        return xyxy.reshape(-1, 4) if n else np.empty((0, 4), dtype=float), confs, cls_ids
    raise TypeError(
        f"unexpected rfdetr predict() return type: {type(preds)!r}"
    )
