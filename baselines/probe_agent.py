"""Heuristic baseline: probe placements and infer distractors from snap feedback.

Strategy (a simple loop + goal + memory pattern):
  * scan a grid of candidate drop points over the central hole region;
  * for each point, try to drop every still-unsolved piece there (`try_fit`);
  * a real fragment eventually snaps home -> mark it real and stop probing it;
  * a distractor never snaps; whatever has not snapped when the goal is met
    (all but n_distractors pieces placed) or the budget runs low = distractor.

This shows the harness is solvable without the answer key, but it is *expensive*
in tool calls and degrades under a tight budget — which is exactly why a
perception/reasoning agent is the better answer. Treat it as a floor for
"real" agents, not a target.
"""
from __future__ import annotations

from typing import List

from harness.agent_base import DistractorAgent
from harness.session import BudgetExceeded, ToolsProxy


def _grid_points(board, frac=0.55, step=22):
    """Grid over the central `frac` of the board (where the hole sits)."""
    bx, by, bw, bh = board["x"], board["y"], board["w"], board["h"]
    cx, cy = bx + bw / 2, by + bh / 2
    hw, hh = bw * frac / 2, bh * frac / 2
    pts = []
    y = cy - hh
    while y <= cy + hh:
        x = cx - hw
        while x <= cx + hw:
            pts.append((x, y))
            x += step
        y += step
    # probe from the centre outwards (real pieces tend to cluster near centre)
    pts.sort(key=lambda p: (p[0] - cx) ** 2 + (p[1] - cy) ** 2)
    return pts


class ProbeAgent(DistractorAgent):
    name = "probe"

    def solve(self, tools: ToolsProxy) -> List[int]:
        task = tools.task()
        n_dist = task["n_distractors"]
        board = task["canvas"]["board"]
        pieces = [p["id"] for p in tools.list_pieces()]

        snapped: set = set()
        active = list(pieces)
        points = _grid_points(board)

        try:
            for (x, y) in points:
                if len(pieces) - len(snapped) <= n_dist:
                    break  # goal: all real pieces accounted for
                for pid in list(active):
                    if tools.calls_used >= tools.budget - 1:
                        raise BudgetExceeded("self-stop before hard cap")
                    res = tools.try_fit(pid, x, y)
                    tools.remember(f"probed_{pid}", tools.recall(f"probed_{pid}", 0) + 1)
                    if res["snapped"]:
                        snapped.add(pid)
                        active.remove(pid)
        except BudgetExceeded:
            pass

        # Distractors = never snapped. If the budget cut us off and too many are
        # unsnapped, fall back to the least-probed-without-snap pieces.
        unsnapped = [p for p in pieces if p not in snapped]
        if len(unsnapped) == n_dist:
            guess = unsnapped
        else:
            unsnapped.sort(key=lambda p: tools.recall(f"probed_{p}", 0), reverse=True)
            guess = unsnapped[:n_dist]
        tools.submit_distractors(guess)
        return guess
