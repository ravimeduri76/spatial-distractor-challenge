"""Spatial distractor-detection challenge harness."""
from .agent_base import DistractorAgent
from .challenge import ChallengeSpec, N_DISTRACTORS, N_PIECES
from .session import BudgetExceeded, PuzzleSession, ToolsProxy

__all__ = [
    "DistractorAgent", "ChallengeSpec", "PuzzleSession", "ToolsProxy",
    "BudgetExceeded", "N_PIECES", "N_DISTRACTORS",
]


def make_bridge(mock: bool = False, headless: bool = True):
    """Factory: real Chromium bridge, or the browserless mock for tests/CI."""
    if mock:
        from .mock_bridge import MockBridge
        return MockBridge()
    from .browser import GameBridge
    return GameBridge(headless=headless)
