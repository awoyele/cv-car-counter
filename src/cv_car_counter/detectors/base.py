"""Detector protocol shared by all backends.

A detector consumes one BGR frame at a time and returns a list of
:class:`Detection` records for the classes the counter cares about (by default
the COCO ``car`` class). Backends are responsible for their own model loading,
class filtering, and confidence thresholding.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable
import importlib


@dataclass(frozen=True)
class Detection:
    """One detected car.

    ``box`` is xyxy in image pixels. ``confidence`` is the detector's score.
    ``class_id`` is the source-model class id (kept for auditability).
    """
    box: tuple[float, float, float, float]
    confidence: float
    class_id: int


@runtime_checkable
class Detector(Protocol):
    name: str
    model_version: str

    def detect(self, frame) -> list[Detection]:  # frame is a numpy BGR array
        ...


_REGISTRY = {
    "yolo": ("cv_car_counter.detectors.yolo", "YoloDetector"),
    "rfdtr": ("cv_car_counter.detectors.rfdtr", "RfdetrDetector"),
}


def get_detector(name: str, **kwargs) -> Detector:
    """Instantiate a registered detector backend by name."""
    try:
        module_name, class_name = _REGISTRY[name]
    except KeyError as e:
        raise ValueError(
            f"unknown detector {name!r}; choose one of {sorted(_REGISTRY)}"
        ) from e
    module = importlib.import_module(module_name)
    cls = getattr(module, class_name)
    return cls(**kwargs)
