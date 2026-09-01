from __future__ import annotations

import pytest

from cv_car_counter.detectors import Detection, get_detector


def test_get_detector_unknown_name():
    with pytest.raises(ValueError, match="unknown detector"):
        get_detector("does-not-exist")


def test_get_detector_yolo_imports_module(monkeypatch):
    # The YoloDetector import-guards ultralytics; constructing without it
    # available should raise ImportError, proving the registry resolves to the
    # right module/class. Force the guard to fire even if ultralytics is
    # installed in the test environment.
    import sys
    import cv_car_counter.detectors.yolo as yolo_mod

    monkeypatch.setitem(sys.modules, "ultralytics", None)

    with pytest.raises(ImportError, match="ultralytics"):
        get_detector("yolo")


def test_get_detector_rfdtr_imports_module(monkeypatch):
    # Force the import-guard even if rfdetr is installed in this environment.
    import sys

    monkeypatch.setitem(sys.modules, "rfdetr", None)

    with pytest.raises(ImportError, match="rfdetr"):
        get_detector("rfdtr")


def test_detection_is_frozen():
    d = Detection(box=(1.0, 2.0, 3.0, 4.0), confidence=0.5, class_id=2)
    with pytest.raises(Exception):
        d.confidence = 0.9  # type: ignore[misc]


def test_cli_weights_mapping_does_not_change_yolo_contract():
    from cv_car_counter.cli import _detector_kwargs

    assert _detector_kwargs("yolo", None) == {}
    assert _detector_kwargs("yolo", "yolov8n.pt") == {"weights": "yolov8n.pt"}
    assert _detector_kwargs("rfdtr", None) == {}
    assert _detector_kwargs("rfdtr", "nano") == {"size": "nano"}
    assert _detector_kwargs("rfdtr", "rfdetr-medium") == {"size": "medium"}
    assert _detector_kwargs("rfdtr", "/tmp/custom.pth") == {"checkpoint": "/tmp/custom.pth"}
