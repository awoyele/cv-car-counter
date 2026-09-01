"""Detector box-format tests that run WITHOUT the real heavy backends.

These prove the detect() box-conversion logic of each detector path by
monkeypatching the backend model, so the code is verified even on machines
where torch/rfdetr can't be installed.
"""

from __future__ import annotations

import numpy as np
import pytest

from cv_car_counter.detectors import Detection


class _Tensor:
    """Minimal torch-tensor shim: .cpu().numpy() returns the wrapped array."""

    def __init__(self, arr):
        self._arr = arr

    def cpu(self):
        return self

    def numpy(self):
        return self._arr

    def __len__(self):
        return len(self._arr)


def test_yolo_detect_box_conversion(monkeypatch):
    """YOLO returns xyxy boxes; detect() must forward them unchanged."""
    ultralytics = pytest.importorskip("ultralytics")
    import cv_car_counter.detectors.yolo.detector as yolo_mod

    class _Boxes:
        def __init__(self, xyxy, conf, cls):
            self.xyxy = _Tensor(xyxy)
            self.conf = _Tensor(conf)
            self.cls = _Tensor(cls)
        def __len__(self):
            return len(self.xyxy)

    class _Result:
        def __init__(self, boxes):
            self.boxes = boxes

    class _FakeYOLO:
        def __init__(self, *a, **k):
            pass
        def predict(self, frame, **k):
            # one car (cls 2), one motorbike (cls 3) that must be filtered out
            return [_Result(
                _Boxes(
                    xyxy=np.array([[10.0, 20.0, 30.0, 40.0]], dtype=float),
                    conf=np.array([0.9], dtype=float),
                    cls=np.array([2], dtype=int),
                )
            )]

    monkeypatch.setattr(ultralytics, "YOLO", _FakeYOLO)
    det = yolo_mod.YoloDetector(weights="fake.pt", confidence=0.3)
    out = det.detect(np.zeros((4, 4, 3), dtype=np.uint8))
    assert len(out) == 1
    d = out[0]
    assert isinstance(d, Detection)
    assert d.box == (10.0, 20.0, 30.0, 40.0)
    assert d.class_id == 2
    assert d.confidence == pytest.approx(0.9)


def test_yolo_detect_filters_non_car_classes(monkeypatch):
    ultralytics = pytest.importorskip("ultralytics")
    import cv_car_counter.detectors.yolo.detector as yolo_mod

    class _Boxes:
        def __init__(self, xyxy, conf, cls):
            self.xyxy = _Tensor(xyxy)
            self.conf = _Tensor(conf)
            self.cls = _Tensor(cls)
        def __len__(self): return len(self.xyxy)
    class _Result:
        def __init__(self, boxes): self.boxes = boxes
    class _FakeYOLO:
        def __init__(self, *a, **k): pass
        def predict(self, frame, **k):
            return [_Result(_Boxes(
                xyxy=np.array([[0,0,1,1],[2,2,3,3],[4,4,5,5]], dtype=float),
                conf=np.array([0.9,0.8,0.7], dtype=float),
                cls=np.array([2,3,7], dtype=int),  # car, motorbike, truck
            ))]

    monkeypatch.setattr(ultralytics, "YOLO", _FakeYOLO)
    det = yolo_mod.YoloDetector()
    out = det.detect(np.zeros((4, 4, 3), dtype=np.uint8))
    assert len(out) == 1  # only the car survives
    assert out[0].class_id == 2


def test_rfdtr_detect_box_conversion(monkeypatch):
    """RF-DETR predict() returns supervision-style xyxy detections; detect()
    must forward them as xyxy and drop non-car classes. Frames are BGR, so
    detect() must pass RGB (channel-reversed) into predict()."""
    import cv_car_counter.detectors.rfdtr.detector as rfdtr_mod
    import sys

    seen = {}

    class _FakeDetections:
        def __init__(self, xyxy, confidence, class_id):
            self.xyxy = np.array(xyxy, dtype=float)
            self.confidence = np.array(confidence, dtype=float)
            self.class_id = np.array(class_id, dtype=int)

        def __len__(self):
            return len(self.xyxy)

    class _FakeModel:
        def predict(self, frame, threshold=0.3):
            seen["frame"] = frame
            seen["threshold"] = threshold
            return _FakeDetections(
                xyxy=[[10.0, 20.0, 30.0, 40.0]],
                confidence=[0.88],
                class_id=[3],
            )

    class _FakeRFDETRNano:
        def __init__(self, **kwargs):
            seen["init_kwargs"] = kwargs

        def __new__(cls, **kwargs):
            seen["init_kwargs"] = kwargs
            return _FakeModel()

    fake_pkg = type("m", (), {"RFDETRNano": _FakeRFDETRNano})
    monkeypatch.setitem(sys.modules, "rfdetr", fake_pkg)
    det = rfdtr_mod.RfdetrDetector(confidence=0.3)
    bgr = np.zeros((4, 4, 3), dtype=np.uint8)
    bgr[:, :, 0] = 7  # distinctive B channel
    out = det.detect(bgr)
    assert len(out) == 1
    d = out[0]
    assert isinstance(d, Detection)
    assert d.box == (10.0, 20.0, 30.0, 40.0)
    assert d.class_id == 3
    assert d.confidence == pytest.approx(0.88)
    # BGR -> RGB: the distinctive 7 must move from channel 0 to channel 2
    assert seen["frame"][0, 0, 2] == 7
    assert seen["frame"][0, 0, 0] == 0
    assert seen["threshold"] == 0.3


def test_rfdtr_detect_filters_non_car(monkeypatch):
    import cv_car_counter.detectors.rfdtr.detector as rfdtr_mod
    import sys

    class _FakeDetections:
        def __init__(self, xyxy, confidence, class_id):
            self.xyxy = np.array(xyxy, dtype=float)
            self.confidence = np.array(confidence, dtype=float)
            self.class_id = np.array(class_id, dtype=int)

        def __len__(self):
            return len(self.xyxy)

    class _FakeModel:
        def predict(self, frame, threshold=0.3):
            return _FakeDetections(
                xyxy=[[0, 0, 1, 1], [2, 2, 3, 3], [4, 4, 5, 5]],
                confidence=[0.9, 0.8, 0.7],
                class_id=[3, 4, 8],  # car, motorcycle, truck (original COCO ids)
            )

    class _FakeRFDETRNano:
        def __new__(cls, **kwargs):
            return _FakeModel()

    fake_pkg = type("m", (), {"RFDETRNano": _FakeRFDETRNano})
    monkeypatch.setitem(sys.modules, "rfdetr", fake_pkg)
    det = rfdtr_mod.RfdetrDetector()
    out = det.detect(np.zeros((4, 4, 3), dtype=np.uint8))
    assert len(out) == 1
    assert out[0].class_id == 3


def test_rfdtr_unknown_size(monkeypatch):
    import cv_car_counter.detectors.rfdtr.detector as rfdtr_mod
    import sys

    fake_pkg = type("m", (), {"RFDETRNano": object})
    monkeypatch.setitem(sys.modules, "rfdetr", fake_pkg)
    with pytest.raises(ValueError, match="unknown rfdetr size"):
        rfdtr_mod.RfdetrDetector(size="xlarge")
