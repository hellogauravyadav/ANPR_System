"""End-to-end ANPR: detect -> track -> (throttled) OCR -> vote -> results with confidences."""
from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np

from .config import load_config, resolve
from .ocr import OCRRead, crop_with_padding, sharpness
from .tracks import PlateTrack, Read, TrackBook


@dataclass
class PlateResult:
    box: tuple[int, int, int, int]
    det_conf: float           # YOLO box confidence
    text: str                 # "" while still unread
    ocr_conf: float           # OCR confidence after voting
    score: float              # combined confidence 0..1  (sqrt(det * ocr))
    valid: bool               # text matches an Indian plate layout
    locked: bool              # stable result, OCR paused for this plate
    track_id: int | None = None


class ANPRPipeline:
    """`detector` / `ocr` can be injected (used by the unit tests); otherwise built from config."""

    def __init__(self, cfg: SimpleNamespace | None = None, detector=None, ocr=None):
        self.cfg = cfg or load_config()
        d, o, t = self.cfg.detector, self.cfg.ocr, self.cfg.tracking

        if detector is None:
            from .detector import PlateDetector
            detector = PlateDetector(resolve(d.weights), d.conf, d.iou, d.imgsz, d.device,
                                     d.half, d.max_det, d.tracker)
        self.detector = detector

        if ocr is None and o.enabled:
            from .ocr import PlateOCR
            ocr = PlateOCR(o.languages, o.gpu, o.min_text_conf)
        self.ocr = ocr

        self.book = TrackBook(t.max_history, t.stale_after, o.lock_votes, o.lock_conf)
        self._ocr_calls = 0

    # ------------------------------------------------------------------
    def _read_crop(self, frame, box, force: bool) -> tuple[Read | None, np.ndarray | None]:
        o = self.cfg.ocr
        crop = crop_with_padding(frame, box, o.pad)
        if crop is None:
            return None, None
        if not force and o.min_sharpness > 0:
            import cv2
            if sharpness(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)) < o.min_sharpness:
                return None, None
        result: OCRRead | None = self.ocr.read(crop)
        self._ocr_calls += 1
        if result is None:
            return None, crop
        return Read(result.text, result.conf, result.valid, result.fixes), crop

    def _wants_ocr(self, tr: PlateTrack, frame_idx: int) -> bool:
        o = self.cfg.ocr
        if self.ocr is None or tr.locked:
            return False
        x1, _, x2, _ = tr.box
        if (x2 - x1) < o.min_plate_width:
            return False
        return frame_idx - tr.last_ocr_frame >= o.every_n_frames

    @staticmethod
    def _result(tr: PlateTrack, track_id) -> PlateResult:
        return PlateResult(tr.box, tr.det_conf, tr.text, tr.ocr_conf if tr.text else 0.0,
                           tr.score, tr.valid, tr.locked, track_id)

    # ------------------------------------------------------------------
    def process_frame(self, frame: np.ndarray, frame_idx: int = 0) -> list[PlateResult]:
        """Video / webcam: tracking + throttled OCR + temporal voting."""
        dets = self.detector.track(frame)
        pairs: list[tuple[PlateTrack, int | None]] = []
        for k, d in enumerate(dets):
            tid = d.track_id if d.track_id is not None else -(k + 1)   # untracked fallback
            tr = self.book.get(tid, frame_idx)
            tr.update_det(d.conf, d.box, frame_idx)
            pairs.append((tr, d.track_id))

        todo = [tr for tr, _ in pairs if self._wants_ocr(tr, frame_idx)]
        todo.sort(key=lambda t: t.area, reverse=True)           # biggest (nearest) plates first
        for tr in todo[: self.cfg.ocr.max_per_frame]:
            read, crop = self._read_crop(frame, tr.box, force=False)
            if read is not None:
                tr.add_read(read, frame_idx, crop)
            else:
                tr.last_ocr_frame = frame_idx                    # retry after every_n_frames

        self.book.prune(frame_idx)
        return [self._result(tr, tid) for tr, tid in pairs]

    def process_image(self, frame: np.ndarray) -> list[PlateResult]:
        """Single image: no tracking, OCR every detected plate once (all preprocessing variants)."""
        results = []
        for d in self.detector.predict(frame):
            self._img_counter = getattr(self, "_img_counter", 0) + 1
            tr = PlateTrack(self._img_counter, 0)
            tr.update_det(d.conf, d.box, 0)
            if self.ocr is not None:
                read, crop = self._read_crop(frame, d.box, force=True)
                if read is not None:
                    tr.add_read(read, 0, crop)
            self.book.finished.append(tr)                     # so the logger can pick it up
            results.append(self._result(tr, None))
        return results

    # ------------------------------------------------------------------
    def pop_finished(self) -> list[PlateTrack]:
        return self.book.pop_finished()

    def flush(self) -> list[PlateTrack]:
        return self.book.flush()
