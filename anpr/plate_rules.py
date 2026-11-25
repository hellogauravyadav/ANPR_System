from __future__ import annotations

import re
from dataclasses import dataclass

STATE_CODES = {
    "AN", "AP", "AR", "AS", "BR", "CH", "CG", "DD", "DL", "DN", "GA", "GJ", "HP", "HR",
    "JH", "JK", "KA", "KL", "LA", "LD", "MH", "ML", "MN", "MP", "MZ", "NL", "OD", "OR",
    "PB", "PY", "RJ", "SK", "TN", "TR", "TS", "TG", "UA", "UK", "UP", "WB",
}

_TO_DIGIT = {"O": "0", "Q": "0", "D": "0", "I": "1", "L": "1", "Z": "2",
             "S": "5", "G": "6", "B": "8", "T": "7", "A": "4"}
_TO_LETTER = {"0": "O", "1": "I", "2": "Z", "5": "S", "6": "G", "8": "B", "4": "A"}

_BH_RE = re.compile(r"^\d{2}BH\d{4}[A-Z]{1,2}$")
_COST_UNFIXABLE = 10


@dataclass(frozen=True)
class PlateText:
    text: str           # cleaned / corrected text
    valid: bool         # matches an Indian plate layout with a known state code
    state: str = ""
    fixes: int = 0      # number of characters changed by the repair step


def clean(text: str) -> str:
    """Upper-case and keep only A-Z / 0-9."""
    return re.sub(r"[^A-Z0-9]", "", (text or "").upper())


def _fit(ch: str, want: str) -> tuple[str, int]:
    """Make `ch` the wanted type ('L' letter / 'D' digit). Returns (char, cost)."""
    if want == "D":
        if ch.isdigit():
            return ch, 0
        if ch in _TO_DIGIT:
            return _TO_DIGIT[ch], 1
    else:
        if ch.isalpha():
            return ch, 0
        if ch in _TO_LETTER:
            return _TO_LETTER[ch], 1
    return ch, _COST_UNFIXABLE


def _try_layout(text: str, rto_len: int) -> tuple[str, int] | None:
    n = len(text)
    series = n - 2 - rto_len - 4
    if not 0 <= series <= 3:
        return None
    kinds = "L" * 2 + "D" * rto_len + "L" * series + "D" * 4
    out, cost = [], 0
    for ch, kind in zip(text, kinds):
        new, c = _fit(ch, kind)
        out.append(new)
        cost += c
    return "".join(out), cost


def correct(raw: str) -> PlateText:
    """Clean the text, repair look-alike characters by position and validate."""
    text = clean(raw)
    # An "IND" strip can get glued to the front of the read.
    if text.startswith("IND") and len(text) >= 11:
        text = text[3:]
    if not 8 <= len(text) <= 11:
        return PlateText(text, False)

    # Bharat (BH) series
    if len(text) in (10, 11) and text[2:4] == "BH":
        kinds = "DD" + "LL" + "DDDD" + "L" * (len(text) - 8)
        out, cost = [], 0
        for ch, kind in zip(text, kinds):
            new, c = _fit(ch, kind)
            out.append(new)
            cost += c
        cand = "".join(out)
        if cost < _COST_UNFIXABLE and _BH_RE.match(cand):
            return PlateText(cand, True, "BH", cost)

    best: tuple[str, int] | None = None
    best_rank = 0
    for rto_len in (2, 1):
        cand = _try_layout(text, rto_len)
        if cand is None:
            continue
        rank = cand[1] + (2 if rto_len == 1 else 0)     # 1-digit RTOs are rare: prefer the 2-digit layout
        if best is None or rank < best_rank:
            best, best_rank = cand, rank
    if best is None:
        return PlateText(text, False)

    cand, cost = best
    state = cand[:2]
    valid = cost < _COST_UNFIXABLE and state in STATE_CODES
    if not valid and cost < _COST_UNFIXABLE:
        # State code might itself be mis-read (e.g. R1 -> RJ is not fixable by type alone).
        return PlateText(cand, False, state, cost)
    if not valid:
        return PlateText(text, False)
    return PlateText(cand, True, state, cost)


def pretty(text: str, valid: bool) -> str:
    """RJ14AB1234 -> 'RJ 14 AB 1234' (only if the text is a valid plate)."""
    if not valid:
        return text
    if len(text) >= 10 and text[2:4] == "BH":
        return f"{text[:2]} BH {text[4:8]} {text[8:]}"
    m = re.match(r"^([A-Z]{2})(\d{1,2})([A-Z]{0,3})(\d{4})$", text)
    if not m:
        return text
    return " ".join(part for part in m.groups() if part)
