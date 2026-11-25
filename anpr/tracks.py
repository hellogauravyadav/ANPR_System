from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass

import numpy as np

from .plate_rules import correct


@dataclass
class Read:
    text: str
    conf: float
    valid: bool
    fixes: int = 0


class PlateTrack:
    def __init__(self, track_id: int, frame_idx: int, max_history=15,
                 lock_votes=3, lock_conf=0.60):
        self.id = track_id
        self.first_frame = self.last_seen = frame_idx
        self.last_ocr_frame = -10**9
        self.det_conf = 0.0
        self.box = (0, 0, 0, 0)
        self.reads: deque[Read] = deque(maxlen=max_history)
        self.text = ""
        self.ocr_conf = 0.0
        self.valid = False
        self.locked = False
        self.lock_area = 0
        self.best_crop: np.ndarray | None = None
        self._best_crop_conf = -1.0
        self.lock_votes, self.lock_conf = lock_votes, lock_conf

    # ---- detector side ------------------------------------------------
    @property
    def area(self) -> int:
        x1, y1, x2, y2 = self.box
        return max(0, x2 - x1) * max(0, y2 - y1)

    def update_det(self, conf: float, box, frame_idx: int) -> None:
        self.det_conf = conf if self.det_conf == 0 else 0.7 * self.det_conf + 0.3 * conf
        self.box, self.last_seen = box, frame_idx
        if self.locked and self.area > 1.6 * self.lock_area:   # plate got much bigger -> re-read
            self.locked = False

    # ---- OCR side -----------------------------------------------------
    def add_read(self, read: Read, frame_idx: int, crop: np.ndarray | None = None) -> None:
        self.last_ocr_frame = frame_idx
        self.reads.append(read)
        self._vote()
        if crop is not None and read.conf * (2 if read.valid else 1) > self._best_crop_conf:
            self._best_crop_conf = read.conf * (2 if read.valid else 1)
            self.best_crop = crop
        recent = list(self.reads)[-self.lock_votes:]
        if any(r.fixes for r in recent):          # repaired reads need extra agreement before locking
            recent = list(self.reads)[-(self.lock_votes + 2):]
            need = self.lock_votes + 2
        else:
            need = self.lock_votes
        if (len(recent) == need and self.valid and self.ocr_conf >= self.lock_conf
                and all(r.text == self.text for r in recent)):
            self.locked, self.lock_area = True, self.area

    def _vote(self) -> None:
        reads = list(self.reads)
        weight_by_len: dict[int, float] = defaultdict(float)
        for r in reads:
            weight_by_len[len(r.text)] += r.conf * (1.5 if r.valid else 1.0)
        length = max(weight_by_len, key=weight_by_len.get)
        sel = [r for r in reads if len(r.text) == length]

        chars, agree = [], []
        for i in range(length):
            w: dict[str, float] = defaultdict(float)
            for r in sel:
                w[r.text[i]] += r.conf * (1.5 if r.valid else 1.0)
            winner = max(w, key=w.get)
            chars.append(winner)
            agree.append(w[winner] / sum(w.values()))
        voted = correct("".join(chars))
        mean_conf = sum(r.conf for r in sel) / len(sel)
        self.text = voted.text
        self.valid = voted.valid
        self.ocr_conf = mean_conf * (0.5 + 0.5 * float(np.mean(agree)))

    # ---- combined score -----------------------------------------------
    @property
    def score(self) -> float:
        """Geometric mean of detector and OCR confidence (or detector only if unread)."""
        if not self.text:
            return self.det_conf
        return float((self.det_conf * self.ocr_conf) ** 0.5)


class TrackBook:
    """Container of active tracks; closes tracks that have not been seen for `stale_after` frames."""

    def __init__(self, max_history=15, stale_after=30, lock_votes=3, lock_conf=0.60):
        self.tracks: dict[int, PlateTrack] = {}
        self.finished: list[PlateTrack] = []
        self.kw = dict(max_history=max_history, lock_votes=lock_votes, lock_conf=lock_conf)
        self.stale_after = stale_after

    def get(self, track_id: int, frame_idx: int) -> PlateTrack:
        if track_id not in self.tracks:
            self.tracks[track_id] = PlateTrack(track_id, frame_idx, **self.kw)
        return self.tracks[track_id]

    def prune(self, frame_idx: int) -> None:
        for tid in [t for t, tr in self.tracks.items() if frame_idx - tr.last_seen > self.stale_after]:
            self.finished.append(self.tracks.pop(tid))

    def pop_finished(self) -> list[PlateTrack]:
        out, self.finished = self.finished, []
        return out

    def flush(self) -> list[PlateTrack]:
        """Close everything (call at the end of a video)."""
        self.finished.extend(self.tracks.values())
        self.tracks = {}
        return self.pop_finished()
