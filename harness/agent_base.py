"""The interface every candidate agent implements.

Keep it this small on purpose: the harness does not care which framework you use
(ADK, LangGraph, Deep Agents, or hand-rolled). It only needs a ``solve`` method
that drives a :class:`PuzzleSession` and returns the predicted distractor ids.
"""
from __future__ import annotations

from typing import List

from .session import ToolsProxy


class DistractorAgent:
    #: Shown in result tables.
    name = "unnamed"

    #: If True, the evaluator keeps ONE agent instance across all puzzles, so the
    #: agent may carry learning/memory between episodes (stateful execution).
    #: If False, a fresh instance is created per puzzle (stateless execution).
    STATEFUL = False

    def solve(self, tools: ToolsProxy) -> List[int]:
        """Inspect the puzzle via ``tools`` (a locked :class:`ToolsProxy`) and
        return the 2 distractor ids.

        ``tools`` exposes only the sanctioned tools — there is no path to the
        browser bridge, settings, seed, or answer key. You may also call
        ``tools.submit_distractors([...])`` yourself; if you return a list and
        never submit, the evaluator submits your return value.
        """
        raise NotImplementedError
