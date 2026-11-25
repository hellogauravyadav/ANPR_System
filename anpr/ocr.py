from __future__ import annotations

from dataclasses import dataclass
from statistics import median

import cv2
import numpy as np

from .plate_rules import PlateText, clean, correct

ALLOWLIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
TARGET_HEIGHT = 64        


@dataclass
class OCRRead:
    text: str            # corrected text
    conf: float          # mean token confidence 0..1
    valid: bool          # matches an Indian plate layout
    raw: str = ""        # text before correction (debugging)
    fixes: int = 0       # characters changed by the Indian-format repair


def sharpness(gray: np.ndarray) -> float:
    """Variance of the Laplacian: low = blurry."""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def crop_with_padding(frame: np.ndarray, box, pad: float) -> np.ndarray | None:
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = box
    px, py = int((x2 - x1) * pad), int((y2 - y1) * pad)
    x1, y1 = max(0, x1 - px), max(0, y1 - py)
    x2, y2 = min(w, x2 + px), min(h, y2 + py)
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None
    return frame[y1:y2, x1:x2].copy()


def _upscale(img: np.ndarray) -> np.ndarray:
    h = img.shape[0]
    if h >= TARGET_HEIGHT:
        return img
    scale = TARGET_HEIGHT / h
    return cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)


def preprocess_variants(crop_bgr: np.ndarray):
    """Yield image variants, best-first. Later ones are only tried if the first fails."""
    big = _upscale(crop_bgr)
    gray = cv2.cvtColor(big, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4)).apply(gray)
    yield clahe
    yield big                                            # untouched colour image
    blur = cv2.GaussianBlur(gray, (0, 0), 1.0)
    yield cv2.addWeighted(gray, 1.6, blur, -0.6, 0)      # unsharp mask


def assemble_tokens(tokens) -> tuple[str, float]:
    """Merge EasyOCR tokens into one string; groups rows so 2-line plates read top->bottom."""
    items = []
    for bbox, text, conf in tokens:
        t = clean(text)
        if not t or t == "IND":
            continue
        xs = [p[0] for p in bbox]
        ys = [p[1] for p in bbox]
        items.append({"t": t, "c": float(conf), "x": min(xs),
                      "y": (min(ys) + max(ys)) / 2, "h": max(ys) - min(ys)})
    if not items:
        return "", 0.0

    med_h = max(median(i["h"] for i in items), 1.0)
    rows: list[list[dict]] = []
    for it in sorted(items, key=lambda i: i["y"]):
        if rows and abs(it["y"] - np.mean([r["y"] for r in rows[-1]])) < 0.6 * med_h:
            rows[-1].append(it)
        else:
            rows.append([it])

    ordered = [it for row in rows for it in sorted(row, key=lambda i: i["x"])]
    text = "".join(i["t"] for i in ordered)
    total = sum(len(i["t"]) for i in ordered)
    conf = sum(i["c"] * len(i["t"]) for i in ordered) / total
    return text, conf


class PlateOCR:
    def __init__(self, languages=("en",), gpu="auto", min_text_conf=0.25):
        import easyocr
        import torch

        if gpu == "auto":
            use_gpu = torch.cuda.is_available()
        else:
            use_gpu = str(gpu).lower() in ("1", "true", "yes")
        self.reader = easyocr.Reader(list(languages), gpu=use_gpu, verbose=False)
        self.min_text_conf = min_text_conf

    def _read_once(self, img: np.ndarray) -> OCRRead | None:
        tokens = self.reader.readtext(img, allowlist=ALLOWLIST, detail=1, paragraph=False,
                                      batch_size=1)
        raw, conf = assemble_tokens(tokens)
        if not raw or conf < self.min_text_conf:
            return None
        fixed: PlateText = correct(raw)
        conf *= 0.92 ** fixed.fixes          # every repaired character makes the read less certain
        return OCRRead(text=fixed.text, conf=conf, valid=fixed.valid, raw=raw, fixes=fixed.fixes)

    def read(self, crop_bgr: np.ndarray) -> OCRRead | None:
        """Try preprocessing variants until one gives a valid Indian plate; else best guess."""
        best: OCRRead | None = None
        for variant in preprocess_variants(crop_bgr):
            read = self._read_once(variant)
            if read is None:
                continue
            if read.valid and read.conf >= 0.5:
                return read
            if best is None or (read.valid, read.conf) > (best.valid, best.conf):
                best = read
        return best
