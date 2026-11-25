"""Pipeline tests with a fake detector + fake OCR (no model weights or GPU needed)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from anpr.config import load_config
from anpr.detector import Detection
from anpr.ocr import OCRRead, assemble_tokens
from anpr.pipeline import ANPRPipeline
from anpr.tracks import PlateTrack, Read
from anpr.visualize import draw_results


class FakeDetector:
    device = "cpu"

    def __init__(self):
        self.i = 0

    def track(self, frame):
        self.i += 1
        return [Detection((100, 100, 260, 150), 0.90, 7)]

    def predict(self, frame):
        return [Detection((100, 100, 260, 150), 0.88, None)]


class FakeOCR:
    """Noisy reader: wrong on every 3rd call."""

    def __init__(self):
        self.n = 0

    def read(self, crop):
        self.n += 1
        text = "RJ14AB1234" if self.n % 3 else "RJ14A81Z34"
        return OCRRead(text, 0.9, True) if self.n % 3 else OCRRead(text, 0.5, False)


def _cfg():
    cfg = load_config()
    cfg.ocr.min_sharpness = 0       # synthetic frames are flat
    cfg.ocr.every_n_frames = 1
    return cfg


def test_voting_fixes_noisy_reads_and_locks():
    pipe = ANPRPipeline(_cfg(), detector=FakeDetector(), ocr=FakeOCR())
    frame = np.full((300, 400, 3), 120, np.uint8)
    res = None
    for i in range(10):
        res = pipe.process_frame(frame, i)[0]
    assert res.text == "RJ14AB1234" and res.valid
    assert 0 < res.score <= 1 and res.ocr_conf > 0.5
    assert res.track_id == 7


def test_ocr_stops_after_lock():
    ocr = FakeOCR()
    ocr.read = lambda crop, _c=[0]: (_c.__setitem__(0, _c[0] + 1), OCRRead("KA41EN1051", 0.95, True))[1]
    pipe = ANPRPipeline(_cfg(), detector=FakeDetector(), ocr=ocr)
    frame = np.full((300, 400, 3), 120, np.uint8)
    for i in range(20):
        r = pipe.process_frame(frame, i)[0]
    assert r.locked
    # locked after lock_votes (3) reads; no further OCR calls
    assert pipe._ocr_calls == 3


def test_image_mode_and_logging_hand_off():
    pipe = ANPRPipeline(_cfg(), detector=FakeDetector(), ocr=FakeOCR())
    frame = np.full((300, 400, 3), 120, np.uint8)
    res = pipe.process_image(frame)
    assert len(res) == 1 and res[0].text
    assert len(pipe.pop_finished()) == 1


def test_stale_tracks_are_finished():
    cfg = _cfg(); cfg.ocr.enabled = False
    pipe = ANPRPipeline(cfg, detector=FakeDetector(), ocr=None)
    frame = np.zeros((300, 400, 3), np.uint8)
    pipe.process_frame(frame, 0)
    pipe.detector.track = lambda f: []
    pipe.process_frame(frame, 100)
    assert len(pipe.pop_finished()) == 1


def test_draw_does_not_crash_and_changes_pixels():
    pipe = ANPRPipeline(_cfg(), detector=FakeDetector(), ocr=FakeOCR())
    frame = np.full((300, 400, 3), 120, np.uint8)
    res = pipe.process_frame(frame, 0)
    out = draw_results(frame, res, pipe.cfg.display, fps=12.3)
    assert out.shape == frame.shape and (out != frame).any()


def test_assemble_two_row_plate():
    box = lambda x, y: [[x, y], [x + 40, y], [x + 40, y + 20], [x + 0, y + 20]]
    tokens = [(box(60, 5), "41", 0.9), (box(0, 5), "KA", 0.8),      # top row, given out of order
              (box(0, 40), "EN", 0.9), (box(50, 40), "1051", 0.95)]  # bottom row
    text, conf = assemble_tokens(tokens)
    assert text == "KA41EN1051" and 0 < conf <= 1


def test_track_vote_prefers_majority():
    tr = PlateTrack(1, 0)
    for t in ["KA41EN1051", "KA41EN1O51", "KA41EN1051", "KA4IEN1051"]:
        tr.add_read(Read(t, 0.8, True), 0)
    assert tr.text == "KA41EN1051"
