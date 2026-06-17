"""The governed tool surface an agent uses to solve one puzzle.

A ``PuzzleSession`` wraps a single puzzle (one seed). It exposes a small, honest
set of tools and meters every call against a budget. Candidates wrap these
methods as tools in their framework of choice (ADK / LangGraph / Deep Agents).

Tools never reveal the answer key: in vision mode the agent sees pixels and
addressable piece handles (id + current position), but not ``isDistractor``,
``home`` targets, or vertices. The only feedback channel beyond pixels is
``try_fit`` — a real fragment can snap into place, a distractor never will.
"""
from __future__ import annotations

import io
import time
from typing import Any, Dict, List, Optional

from .challenge import ChallengeSpec


class BudgetExceeded(RuntimeError):
    pass


class PuzzleSession:
    def __init__(self, bridge, spec: ChallengeSpec, seed: int,
                 vision: bool = True, enforce_budget: bool = True):
        self.bridge = bridge
        self.spec = spec
        self.seed = seed
        self.vision = vision
        self.enforce_budget = enforce_budget

        self.budget = spec.budget
        self.calls_used = 0
        self.over_budget = False
        self.submission: Optional[List[int]] = None
        self._memory: Dict[str, Any] = {}
        self._board_png: Optional[bytes] = None
        self.trace: List[Dict[str, Any]] = []   # one record per tool call
        self._t0 = time.time()

        bridge.set_settings({**spec.settings, "vision": vision})
        bridge.set_seed(seed)
        bridge.new_game()

    # ---- session-history trace ----
    def _emit(self, tool: str, args: Dict[str, Any], result: Dict[str, Any],
              cost: int) -> None:
        """Append a uniform, framework-agnostic record of a tool call.

        Image bytes are summarised (length only) so the trace stays small and
        readable. This is the session history graders read regardless of which
        agent framework produced it.
        """
        self.trace.append({
            "seq": len(self.trace),
            "t": round(time.time() - self._t0, 4),
            "seed": self.seed,
            "tool": tool,
            "args": args,
            "result": result,
            "cost": cost,
            "calls_used": self.calls_used,
        })

    # ---- budget accounting (governance) ----
    def _charge(self, cost: int = 1) -> None:
        self.calls_used += cost
        if self.enforce_budget and self.calls_used > self.budget:
            self.over_budget = True
            self._emit("_budget_exceeded", {}, {"budget": self.budget}, 0)
            raise BudgetExceeded(
                f"budget {self.budget} exceeded ({self.calls_used} units used)")

    # ---- read-only tools ----
    def task(self) -> Dict[str, Any]:
        """The task card (goal, counts, budget) plus board geometry. Free."""
        card = self.spec.task_card()
        obs = self.bridge.get_observation()
        card["canvas"] = obs.get("canvas")  # {w, h, board, tray} — dims only, no answer key
        self._emit("task", {}, {"n_pieces": card.get("n_pieces")}, 0)
        return card

    def list_pieces(self) -> List[Dict[str, Any]]:
        """Addressable handles for every tray piece: id, current x/y, placed."""
        self._charge(1)
        obs = self.bridge.get_observation()
        pieces = [{"id": p["id"], "x": p["current"]["x"], "y": p["current"]["y"],
                   "placed": p["placed"]} for p in obs["pieces"]]
        self._emit("list_pieces", {}, {"ids": [p["id"] for p in pieces]}, 1)
        return pieces

    def board_image_png(self) -> bytes:
        """PNG bytes of the whole board (hole silhouette + tray)."""
        self._charge(1)
        obs = self.bridge.get_observation()
        self._board_png = self.bridge.b64_to_bytes(self.bridge.dataurl_to_b64(obs["image"]))
        self._emit("board_image_png", {}, {"bytes": len(self._board_png)}, 1)
        return self._board_png

    def piece_image_png(self, piece_id: int, window: int = 130) -> bytes:
        """A square crop of the board centred on one piece (a 'zoom' tool)."""
        self._charge(1)
        obs = self.bridge.get_observation()
        png = self.bridge.b64_to_bytes(self.bridge.dataurl_to_b64(obs["image"]))
        pc = next((p for p in obs["pieces"] if p["id"] == piece_id), None)
        if pc is None:
            raise ValueError(f"no piece {piece_id}")
        out = png
        try:
            from PIL import Image  # optional; real-browser path only
            img = Image.open(io.BytesIO(png)).convert("RGBA")
            cx, cy = pc["current"]["x"], pc["current"]["y"]
            half = window / 2
            box = (max(0, int(cx - half)), max(0, int(cy - half)),
                   min(img.width, int(cx + half)), min(img.height, int(cy + half)))
            buf = io.BytesIO()
            img.crop(box).save(buf, format="PNG")
            out = buf.getvalue()
        except Exception:
            # Pillow missing or placeholder image (mock) — return the full frame.
            out = png
        self._emit("piece_image_png", {"piece_id": piece_id, "window": window},
                   {"bytes": len(out)}, 1)
        return out

    # ---- active probe tool ----
    def try_fit(self, piece_id: int, x: float, y: float, cost: int = 1) -> Dict[str, Any]:
        """Attempt to drop a piece at (x, y). Returns whether it snapped home.

        Real fragments snap when positioned correctly; distractors never snap.
        """
        self._charge(cost)
        res = self.bridge.apply_action({"type": "move", "id": piece_id,
                                        "x": x, "y": y, "cost": cost})
        out = {"piece_id": piece_id, "snapped": bool(res.get("snapped")),
               "placed_count": res.get("state", {}).get("placedCount")}
        self._emit("try_fit", {"piece_id": piece_id, "x": round(x, 1), "y": round(y, 1)},
                   {"snapped": out["snapped"]}, cost)
        return out

    # ---- memory tools (optional helpers; frameworks may use their own) ----
    def remember(self, key: str, value: Any) -> None:
        self._memory[key] = value
        self._emit("remember", {"key": key}, {}, 0)

    def recall(self, key: str, default: Any = None) -> Any:
        hit = key in self._memory
        self._emit("recall", {"key": key}, {"hit": hit}, 0)
        return self._memory.get(key, default)

    # ---- terminal ----
    def submit_distractors(self, ids: List[int]) -> Dict[str, Any]:
        """Submit the predicted distractor ids and end the episode."""
        self.submission = sorted(int(i) for i in ids)
        self._emit("submit_distractors", {"ids": self.submission},
                   {"calls_used": self.calls_used}, 0)
        return {"accepted": True, "submitted": self.submission,
                "calls_used": self.calls_used}

    # ---- the object handed to candidate agents ----
    def tools(self) -> "ToolsProxy":
        """Return the locked tool surface an agent is allowed to use.

        The agent never receives the PuzzleSession itself, so it cannot reach
        ``bridge``, ``spec``, ``seed``, or the answer key.
        """
        return ToolsProxy(self)

    # ---- grading hook (NOT a tool — used by the evaluator only) ----
    def _ground_truth_distractors(self) -> List[int]:
        state = self.bridge.get_state(vision=False)
        return sorted(p["id"] for p in state["pieces"] if p.get("isDistractor"))


class ToolsProxy:
    """The ONLY interface a candidate agent is given.

    It forwards the sanctioned tools and nothing else. The underlying session,
    browser bridge, settings, seed, and ground-truth answer key are not exposed
    through the supported attribute surface. Reaching past this proxy — e.g.
    ``object.__getattribute__(tools, "_ToolsProxy__s")``, ``.bridge``,
    ``get_state(vision=False)``, ``_ground_truth_distractors`` — is an automatic
    disqualifier and is rejected by the grader's static check.
    """

    __slots__ = ("_ToolsProxy__s",)
    _PUBLIC_NAMES = (
        "task", "list_pieces", "board_image_png", "piece_image_png", "try_fit",
        "remember", "recall", "submit_distractors", "calls_used", "budget",
    )
    _BLOCKED_NAMES = frozenset({
        "_ToolsProxy__s", "__dict__", "bridge", "spec", "seed",
        "_ground_truth_distractors", "get_state", "page",
    })

    def __init__(self, session: "PuzzleSession"):
        object.__setattr__(self, "_ToolsProxy__s", session)

    def __getattribute__(self, name: str) -> Any:
        if name in ToolsProxy._BLOCKED_NAMES or name.startswith("_ToolsProxy__"):
            raise AttributeError(name)
        return object.__getattribute__(self, name)

    def __dir__(self) -> List[str]:
        return list(self._PUBLIC_NAMES)

    # --- sanctioned tools (see README tool table) ---
    def task(self) -> Dict[str, Any]:
        return object.__getattribute__(self, "_ToolsProxy__s").task()

    def list_pieces(self) -> List[Dict[str, Any]]:
        return object.__getattribute__(self, "_ToolsProxy__s").list_pieces()

    def board_image_png(self) -> bytes:
        return object.__getattribute__(self, "_ToolsProxy__s").board_image_png()

    def piece_image_png(self, piece_id: int, window: int = 130) -> bytes:
        return object.__getattribute__(self, "_ToolsProxy__s").piece_image_png(
            piece_id, window)

    def try_fit(self, piece_id: int, x: float, y: float, cost: int = 1) -> Dict[str, Any]:
        return object.__getattribute__(self, "_ToolsProxy__s").try_fit(
            piece_id, x, y, cost)

    def remember(self, key: str, value: Any) -> None:
        return object.__getattribute__(self, "_ToolsProxy__s").remember(key, value)

    def recall(self, key: str, default: Any = None) -> Any:
        return object.__getattribute__(self, "_ToolsProxy__s").recall(key, default)

    def submit_distractors(self, ids: List[int]) -> Dict[str, Any]:
        return object.__getattribute__(self, "_ToolsProxy__s").submit_distractors(ids)

    # --- read-only budget meters (governance) ---
    @property
    def calls_used(self) -> int:
        return object.__getattribute__(self, "_ToolsProxy__s").calls_used

    @property
    def budget(self) -> int:
        return object.__getattribute__(self, "_ToolsProxy__s").budget
