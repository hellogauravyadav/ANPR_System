"""Load config.yaml, merge it over safe defaults, and expose it with attribute access."""
from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace

import yaml

ROOT = Path(__file__).resolve().parent.parent

DEFAULTS: dict = {
    "detector": {
        "weights": "weights/best.pt", "conf": 0.35, "iou": 0.50, "imgsz": 640,
        "device": "auto", "half": True, "max_det": 20, "tracker": "bytetrack.yaml",
    },
    "ocr": {
        "enabled": True, "languages": ["en"], "gpu": "auto", "every_n_frames": 5,
        "max_per_frame": 2, "min_plate_width": 40, "pad": 0.08, "min_text_conf": 0.25,
        "min_sharpness": 8.0, "lock_votes": 3, "lock_conf": 0.60,
    },
    "tracking": {"max_history": 15, "stale_after": 30},
    "display": {"show_breakdown": True, "spaced_text": True, "box_thickness": 2, "show_hud": True},
    "output": {"dir": "outputs", "log_csv": True, "save_crops": True, "min_log_score": 0.50},
    "camera": {"width": 1280, "height": 720},
}


def _merge(base: dict, new: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in (new or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def _to_ns(obj):
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: _to_ns(v) for k, v in obj.items()})
    return obj


def resolve(path: str | Path) -> Path:
    """Resolve a path relative to the project root unless it is already absolute."""
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def load_config(path: str | Path | None = None, overrides: dict | None = None) -> SimpleNamespace:
    """Read YAML (if it exists), apply overrides (nested dict), and return a namespace."""
    cfg_path = resolve(path or "config.yaml")
    data: dict = {}
    if cfg_path.exists():
        with open(cfg_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    merged = _merge(DEFAULTS, data)
    merged = _merge(merged, overrides or {})
    return _to_ns(merged)
