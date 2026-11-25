"""CSV log + best-crop saver: one row per finished plate track."""
from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

import cv2

from .tracks import PlateTrack

HEADER = ["timestamp", "source", "track_id", "plate", "valid_format", "det_conf", "ocr_conf",
          "score", "first_frame", "last_frame", "crop_file"]


class PlateLogger:
    def __init__(self, out_dir: Path, source: str, min_score=0.5, save_crops=True, enabled=True):
        self.enabled, self.source = enabled, source
        self.min_score, self.save_crops = min_score, save_crops
        self.out_dir = Path(out_dir)
        self.crop_dir = self.out_dir / "plate_crops"
        self.csv_path = self.out_dir / "plate_log.csv"
        self.count = 0
        if not enabled:
            return
        self.out_dir.mkdir(parents=True, exist_ok=True)
        if save_crops:
            self.crop_dir.mkdir(parents=True, exist_ok=True)
        if not self.csv_path.exists():
            with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(HEADER)

    def log(self, track: PlateTrack) -> bool:
        if not self.enabled or not track.text or track.score < self.min_score:
            return False
        crop_name = ""
        if self.save_crops and track.best_crop is not None:
            crop_name = f"{datetime.now():%Y%m%d_%H%M%S}_{track.id}_{track.text}.jpg"
            cv2.imwrite(str(self.crop_dir / crop_name), track.best_crop)
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                datetime.now().isoformat(timespec="seconds"), self.source, track.id, track.text,
                int(track.valid), f"{track.det_conf:.3f}", f"{track.ocr_conf:.3f}",
                f"{track.score:.3f}", track.first_frame, track.last_seen, crop_name,
            ])
        self.count += 1
        return True
