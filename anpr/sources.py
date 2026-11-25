"""Frame sources: webcam / RTSP (threaded, always newest frame) and video files (every frame)."""
from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import cv2

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".wmv", ".flv"}


def classify_source(src: str) -> str:
    """Return one of: webcam, stream, folder, image, video."""
    if src.isdigit():
        return "webcam"
    if src.lower().startswith(("rtsp://", "http://", "https://", "rtmp://")):
        return "stream"
    p = Path(src)
    if p.is_dir():
        return "folder"
    ext = p.suffix.lower()
    if ext in IMAGE_EXT:
        return "image"
    if ext in VIDEO_EXT:
        return "video"
    raise ValueError(f"Cannot tell what '{src}' is. Use a webcam index, image, folder, video or stream URL.")


class VideoFileCapture:
    """Reads every frame in order (no frame dropping) - best for offline processing."""

    def __init__(self, path: str):
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise FileNotFoundError(f"Could not open video: {path}")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 25.0
        self.size = (int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        self.total = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    def read(self):
        return self.cap.read()

    def release(self):
        self.cap.release()


class ThreadedCapture:
    """Background thread keeps only the latest frame, so slow inference never builds up lag.

    This is the same idea as the asynchronous worker used in the Lab 11 webcam cells.
    """

    def __init__(self, src: str, width: int = 1280, height: int = 720):
        target = int(src) if src.isdigit() else src
        backend = cv2.CAP_DSHOW if (isinstance(target, int) and sys.platform.startswith("win")) else cv2.CAP_ANY
        self.cap = cv2.VideoCapture(target, backend)
        if not self.cap.isOpened():
            raise RuntimeError(
                f"Could not open source '{src}'. Check the index/URL and your OS camera permissions."
            )
        if isinstance(target, int):
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.size = (int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        self.total = 0

        self._lock = threading.Lock()
        self._frame = None
        self._counter = 0
        self._seen = 0
        self._stopped = False
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        while not self._stopped:
            ok, frame = self.cap.read()
            if not ok:
                self._stopped = True
                break
            with self._lock:
                self._frame, self._counter = frame, self._counter + 1

    def read(self, timeout: float = 2.0):
        t0 = time.time()
        while time.time() - t0 < timeout:
            with self._lock:
                if self._counter != self._seen:
                    self._seen = self._counter
                    return True, self._frame.copy()
            if self._stopped:
                break
            time.sleep(0.002)
        return False, None

    def release(self):
        self._stopped = True
        self._thread.join(timeout=1.0)
        self.cap.release()
