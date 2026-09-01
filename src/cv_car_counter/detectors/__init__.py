"""Detector backends.

Two interchangeable detector paths live here:

* :mod:`cv_car_counter.detectors.yolo`  — Ultralytics YOLO (AGPL-3.0 or
  commercial Enterprise license).
* :mod:`cv_car_counter.detectors.rfdtr` — RF-DETR (permissive checkpoint terms).

Both implement the :class:`~cv_car_counter.detectors.base.Detector` protocol so
the rest of the pipeline is detector-agnostic. Select one at runtime with
``--detector yolo`` or ``--detector rfdtr``.
"""

from .base import Detection, Detector, get_detector

__all__ = ["Detection", "Detector", "get_detector"]
