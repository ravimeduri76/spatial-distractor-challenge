"""Ceiling / pipeline sanity check — NOT a real agent.

It reads the answer key (``isDistractor``) from the oracle observation, so it
only works with ``--oracle-vision``. Use it to confirm the harness, scoring, and
seed set are wired correctly (it should score 100%). A submitted agent that
reaches into oracle state this way is disqualified.
"""
from __future__ import annotations

from typing import List

from harness.agent_base import DistractorAgent
from harness.session import PuzzleSession


class OracleAgent(DistractorAgent):
    name = "oracle"
    #: Grader-only: the evaluator hands this agent the full PuzzleSession (not the
    #: locked ToolsProxy) so it can read the answer key. Candidate agents never get
    #: this and cannot reach the answer key through the proxy.
    NEEDS_ORACLE = True

    def solve(self, tools: PuzzleSession) -> List[int]:
        truth = tools._ground_truth_distractors()
        tools.submit_distractors(truth)
        return truth
