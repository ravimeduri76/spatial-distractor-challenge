"""Minimal end-to-end example — your starting point.

It runs against the harness, touches each tool once, and submits a (naive)
answer so you can confirm your setup works:

    python -m harness.evaluate --agent example --mock

Replace the body of `solve` with your real agent. You are encouraged to wrap the
`tools.*` methods as tools in ADK / LangGraph / Deep Agents and let your agent
loop until it is confident, then call `tools.submit_distractors(...)`.
See AGENT_SPEC.md for what we want the agent to demonstrate.
"""
from __future__ import annotations

from typing import List

from harness.agent_base import DistractorAgent
from harness.session import ToolsProxy


class ExampleAgent(DistractorAgent):
    name = "example"
    STATEFUL = False  # set True to keep one instance across all puzzles

    def solve(self, tools: ToolsProxy) -> List[int]:
        task = tools.task()                       # goal + counts + board dims
        pieces = tools.list_pieces()              # [{id, x, y, placed}, ...]

        # A real agent would now look at the picture and the fragments:
        _board_png = tools.board_image_png()      # full board (hole silhouette)
        for p in pieces[:1]:
            _crop = tools.piece_image_png(p["id"])  # zoom on a single fragment
        # ...feed those images to your model / reasoning loop...

        # Naive placeholder: guess the last two ids. (You can do far better.)
        guess = [p["id"] for p in pieces][-task["n_distractors"]:]
        tools.submit_distractors(guess)
        return guess
