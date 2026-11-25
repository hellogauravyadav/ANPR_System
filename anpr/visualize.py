"""Drawing helpers: green plate boxes with a filled label showing text + confidence %."""
from __future__ import annotations

import cv2
import numpy as np

from .plate_rules import pretty

FONT = cv2.FONT_HERSHEY_SIMPLEX
GREEN = (40, 160, 40)        # BGR  valid Indian plate
ORANGE = (0, 140, 255)       # text read but not a valid plate layout
GRAY = (150, 120, 60)        # detected, text not read (yet)
WHITE = (255, 255, 255)


def _color(r) -> tuple[int, int, int]:
    if not r.text:
        return GRAY
    return GREEN if r.valid else ORANGE


def _label(r, spaced: bool) -> str:
    if not r.text:
        return f"plate {r.det_conf * 100:.0f}%"
    text = pretty(r.text, r.valid) if spaced else r.text
    return f"{text}  {r.score * 100:.0f}%"


def draw_results(frame: np.ndarray, results, display, fps: float | None = None) -> np.ndarray:
    out = frame.copy()
    h, w = out.shape[:2]
    fs = float(np.clip(w / 1400, 0.5, 1.0))
    thick = max(1, int(display.box_thickness))
    pad = max(3, int(6 * fs))

    for r in results:
        x1, y1, x2, y2 = r.box
        color = _color(r)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thick)

        label = _label(r, display.spaced_text)
        (tw, th), _ = cv2.getTextSize(label, FONT, fs, 2)
        top = y1 - th - 2 * pad
        if top < 0:                                    # no room above -> put the label below
            top = min(y2, h - th - 2 * pad)
        left = min(x1, max(0, w - tw - 2 * pad))
        cv2.rectangle(out, (left, top), (left + tw + 2 * pad, top + th + 2 * pad), color, -1)
        cv2.putText(out, label, (left + pad, top + th + pad - 1), FONT, fs, WHITE, 2, cv2.LINE_AA)

        if display.show_breakdown and r.text:
            sub = f"det {r.det_conf * 100:.0f}% | ocr {r.ocr_conf * 100:.0f}%" + (" *" if r.locked else "")
            sfs = fs * 0.6
            (sw, sh), _ = cv2.getTextSize(sub, FONT, sfs, 1)
            sy = min(y2 + sh + 6, h - 4)
            cv2.rectangle(out, (x1, sy - sh - 4), (x1 + sw + 6, sy + 4), (30, 30, 30), -1)
            cv2.putText(out, sub, (x1 + 3, sy), FONT, sfs, WHITE, 1, cv2.LINE_AA)

    if display.show_hud:
        hud = f"Plates: {len(results)}" + (f"   FPS: {fps:.1f}" if fps is not None else "")
        cv2.rectangle(out, (0, 0), (int(12 + 11 * len(hud) * fs), int(34 * fs) + 10), (20, 20, 20), -1)
        cv2.putText(out, hud, (8, int(26 * fs) + 4), FONT, 0.75 * fs, WHITE, 2, cv2.LINE_AA)
    return out
