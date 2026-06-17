"""Lower bound: guess two pieces at random.

Expected accuracy = 1 / C(6, 2) = 1/15 ~= 6.7%.
"""
from __future__ import annotations

import random
from typing import List

from harness.agent_base import DistractorAgent
from harness.session import ToolsProxy


class RandomAgent(DistractorAgent):
    name = "random"

    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)

    def solve(self, tools: ToolsProxy) -> List[int]:
        pieces = tools.list_pieces()
        ids = [p["id"] for p in pieces]
        guess = self.rng.sample(ids, tools.task()["n_distractors"])
        tools.submit_distractors(guess)
        return guess
