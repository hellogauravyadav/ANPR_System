from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Detection:
    box: tuple[int, int, int, int]   # x1, y1, x2, y2 in pixels
    conf: float                      # detector confidence 0..1
    track_id: int | None = None


def resolve_device(device: str) -> str:
    if str(device).lower() != "auto":
        return str(device)
    import torch
    return "0" if torch.cuda.is_available() else "cpu"


class PlateDetector:
    def __init__(self, weights: str | Path, conf=0.35, iou=0.5, imgsz=640,
                 device="auto", half=True, max_det=20, tracker="bytetrack.yaml"):
        from ultralytics import YOLO

        weights = Path(weights)
        if not weights.exists():
            raise FileNotFoundError(
                f"Detector weights not found: {weights}\n"
                "Train the model with notebooks/01_train_plate_detector_colab.ipynb and copy "
                "best.pt into the weights/ folder (or pass --weights path/to/best.pt)."
            )
        self.model = YOLO(str(weights))
        self.device = resolve_device(device)
        self.half = bool(half) and self.device != "cpu"
        self.conf, self.iou, self.imgsz = conf, iou, imgsz
        self.max_det, self.tracker = max_det, tracker

    # ------------------------------------------------------------------
    @staticmethod
    def _supports_quantize() -> bool:
        """Newer Ultralytics replaced `half=True` with `quantize=16` (and warns on every frame)."""
        try:
            from ultralytics.cfg import DEFAULT_CFG_DICT
            return "quantize" in DEFAULT_CFG_DICT
        except Exception:
            return False

    def _common(self) -> dict:
        kw = dict(conf=self.conf, iou=self.iou, imgsz=self.imgsz, device=self.device,
                  max_det=self.max_det, verbose=False)
        if self.half:
            if self._supports_quantize():
                kw["quantize"] = 16
            else:
                kw["half"] = True
        return kw

    @staticmethod
    def _parse(result) -> list[Detection]:
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return []
        xyxy = boxes.xyxy.cpu().numpy().astype(int)
        conf = boxes.conf.cpu().numpy()
        ids = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else [None] * len(conf)
        return [Detection(tuple(map(int, b)), float(c), None if i is None else int(i))
                for b, c, i in zip(xyxy, conf, ids)]

    def predict(self, frame: np.ndarray) -> list[Detection]:
        """Stateless detection (single images)."""
        return self._parse(self.model.predict(frame, **self._common())[0])

    def track(self, frame: np.ndarray) -> list[Detection]:
        """Detection + persistent track IDs (video / webcam)."""
        res = self.model.track(frame, persist=True, tracker=self.tracker, **self._common())[0]
        return self._parse(res)

    def reset_tracker(self) -> None:
        """Forget tracker state (call between unrelated videos in one process)."""
        self.model.predictor = None