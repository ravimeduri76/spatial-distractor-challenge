"""Deterministic in-memory stand-in for :class:`GameBridge`.

The real bridge needs headless Chromium (``playwright install chromium``). That is
overkill for unit tests and CI, and unavailable in some sandboxes. ``MockBridge``
implements the exact same ``window.SpatialGame`` seam against a tiny synthetic
puzzle so the tool surface, evaluator, scoring, and probe baseline can be
exercised end-to-end with no browser.

It is intentionally simple: real fragments have a "home" inside the visible hole
and snap when a piece is dropped within tolerance; distractors fit nowhere and
never snap. Images are placeholder PNGs (vision agents need the real bridge).
"""
from __future__ import annotations

import base64
import math
import random
from typing import Any, Dict, List, Optional

# Smallest valid 1x1 PNG (transparent). Real rendering comes from the browser bridge.
_PLACEHOLDER_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
)
_PLACEHOLDER_DATA_URL = "data:image/png;base64," + _PLACEHOLDER_PNG_B64

_W, _H = 900, 820
_BOARD = {"x": 0, "y": 0, "w": _W, "h": 560}
_TRAY = {"x": 0, "y": 560, "w": _W, "h": _H - 560}
_HOLE_CX, _HOLE_CY, _HOLE_R = 450.0, 280.0, 120.0
_SNAP_TOL = 22.0


class MockBridge:
    """Drop-in replacement for GameBridge used by tests / browserless runs."""

    def __init__(self, *args, **kwargs):
        self._settings: Dict[str, Any] = {}
        self._seed: Optional[int] = None
        self._pieces: List[Dict[str, Any]] = []
        self._solved = False
        self._moves = 0
        self._compute = 0

    # --- lifecycle (no-ops for the mock) ---
    def start(self) -> None:  # pragma: no cover - trivial
        pass

    def close(self) -> None:  # pragma: no cover - trivial
        pass

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.close()

    # --- seam wrappers ---
    def set_settings(self, settings: Dict[str, Any]) -> Dict[str, Any]:
        self._settings.update(settings or {})
        return {}

    def set_seed(self, seed) -> Dict[str, Any]:
        self._seed = None if seed is None else int(seed) & 0xFFFFFFFF
        return {}

    def new_game(self) -> Dict[str, Any]:
        rng = random.Random(self._seed if self._seed is not None else random.randrange(1 << 30))
        counts = {"few": 4, "some": 8, "many": 15}
        n_real = counts.get(self._settings.get("count", "few"), 4)
        n_dist = int(self._settings.get("distract", 0) or 0)
        total = n_real + n_dist
        dist_ids = set(rng.sample(range(total), n_dist)) if n_dist else set()

        self._pieces = []
        for pid in range(total):
            is_dist = pid in dist_ids
            if is_dist:
                # decoy fragment: home far outside the visible hole; never snaps
                home = {"x": _HOLE_CX + rng.uniform(220, 300) * rng.choice([-1, 1]),
                        "y": _HOLE_CY + rng.uniform(120, 180) * rng.choice([-1, 1])}
            else:
                a = rng.uniform(0, 2 * math.pi)
                r = _HOLE_R * math.sqrt(rng.uniform(0, 0.7))
                home = {"x": _HOLE_CX + r * math.cos(a), "y": _HOLE_CY + r * math.sin(a)}
            # pieces start scattered in the tray
            cur = {"x": _TRAY["x"] + rng.uniform(0.05, 0.95) * _TRAY["w"],
                   "y": _TRAY["y"] + rng.uniform(0.15, 0.85) * _TRAY["h"]}
            rad = rng.uniform(18, 34)
            self._pieces.append({
                "id": pid, "isDistractor": is_dist, "interior": not is_dist,
                "home": home, "cur": cur, "rot": 0.0, "placed": False, "radius": rad,
            })
        self._solved = False
        self._moves = 0
        self._compute = 0
        return {}

    def _metrics(self) -> Dict[str, Any]:
        real = [p for p in self._pieces if not p["isDistractor"]]
        placed = sum(1 for p in real if p["placed"])
        pos = sum(math.hypot(p["cur"]["x"] - p["home"]["x"], p["cur"]["y"] - p["home"]["y"]) for p in real)
        return {
            "n": len(real), "placed": placed,
            "meanPosErr": pos / (len(real) or 1), "meanAngErr": 0.0,
            "interiorPlaced": placed, "interiorTotal": len(real),
            "distractorsMisplaced": sum(
                1 for p in self._pieces if p["isDistractor"]
                and math.hypot(p["cur"]["x"] - _HOLE_CX, p["cur"]["y"] - _HOLE_CY) < _HOLE_R),
        }

    def get_state(self, vision: bool = False) -> Dict[str, Any]:
        cons = {
            "actions": {"limit": None, "used": self._moves, "enforced": False},
            "compute": {"budget": None, "used": self._compute, "enforced": False},
        }
        if vision:
            return {
                "mode": "vision", "canvas": {"w": _W, "h": _H, "board": _BOARD, "tray": _TRAY},
                "constraints": cons, "solved": self._solved, "failed": False,
                "placedCount": self._metrics()["placed"],
                "pieces": [{"id": p["id"], "placed": p["placed"],
                            "current": {"x": round(p["cur"]["x"], 1), "y": round(p["cur"]["y"], 1)},
                            "rot": round(p["rot"], 3)} for p in self._pieces],
            }
        return {
            "mode": "oracle", "canvas": {"w": _W, "h": _H, "board": _BOARD, "tray": _TRAY},
            "settings": dict(self._settings), "constraints": cons,
            "cutout": [], "solved": self._solved, "failed": False,
            "metrics": self._metrics(),
            "pieces": [{"id": p["id"], "placed": p["placed"], "isDistractor": p["isDistractor"],
                        "interior": p["interior"],
                        "home": {"x": round(p["home"]["x"], 2), "y": round(p["home"]["y"], 2)},
                        "current": {"x": round(p["cur"]["x"], 2), "y": round(p["cur"]["y"], 2)},
                        "rot": round(p["rot"], 3), "vertices": []} for p in self._pieces],
        }

    def get_observation(self) -> Dict[str, Any]:
        return {
            "image": _PLACEHOLDER_DATA_URL,
            "canvas": {"w": _W, "h": _H, "board": _BOARD, "tray": _TRAY},
            "pieces": [{"id": p["id"], "placed": p["placed"],
                        "current": {"x": round(p["cur"]["x"], 1), "y": round(p["cur"]["y"], 1)},
                        "rot": round(p["rot"], 3)} for p in self._pieces],
            "solved": self._solved, "failed": False,
        }

    def apply_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        pid = action.get("id")
        pc = next((p for p in self._pieces if p["id"] == pid), None)
        if pc is None:
            return {"ok": False, "reason": "bad_id", "state": self.get_state()}
        if isinstance(action.get("cost"), (int, float)):
            self._compute += action["cost"]
        if isinstance(action.get("x"), (int, float)):
            pc["cur"]["x"] = action["x"]
        if isinstance(action.get("y"), (int, float)):
            pc["cur"]["y"] = action["y"]
        self._moves += 1
        snapped = False
        if action.get("type") in ("move", "place") and not pc["isDistractor"]:
            d = math.hypot(pc["cur"]["x"] - pc["home"]["x"], pc["cur"]["y"] - pc["home"]["y"])
            if d < _SNAP_TOL:
                pc["placed"] = True
                pc["cur"] = dict(pc["home"])
                snapped = True
        real = [p for p in self._pieces if not p["isDistractor"]]
        if real and all(p["placed"] for p in real):
            self._solved = True
        return {"ok": True, "snapped": snapped, "state": self.get_state()}

    @staticmethod
    def dataurl_to_b64(data_url: str) -> str:
        if data_url and "," in data_url:
            return data_url.split(",", 1)[1]
        return data_url or ""

    @staticmethod
    def b64_to_bytes(b64: str) -> bytes:
        return base64.b64decode(b64) if b64 else b""
