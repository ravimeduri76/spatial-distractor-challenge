"""Fixed challenge configuration.

The task is deliberately constrained so candidates spend their time on *agent
design*, not on tuning the environment:

  * 4 real fragments cut from the picture  (settings ``count="few"``)
  * 2 distractor fragments that fit nowhere (settings ``distract=2``)
  * no rotation                            (settings ``rotation=False``)

The agent must decide which 2 of the 6 tray pieces are the distractors.
Every run is reproducible from its seed, so candidate and reference agents are
graded on the identical set of puzzles.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

# Settings handed to window.SpatialGame.setSettings for every puzzle.
PUZZLE_SETTINGS: Dict[str, object] = {
    "count": "few",        # 4 real interior fragments
    "distract": 2,         # 2 decoy fragments
    "rotation": False,     # translation only — no rotation
    "snap": True,
    "content": "geometric",
    "texture": "rich",
    "size": "medium",
    "cutIrr": "medium",
    "symmetry": "none",
    "pieceIrr": "medium",
}

N_PIECES = 6
N_DISTRACTORS = 2

# Per-episode resource budget (governance). Each tool call costs 1 unit unless a
# heavier `cost` is supplied; exceeding the budget ends the episode as over-budget.
DEFAULT_BUDGET = 60

# PUBLIC SAMPLE SEEDS — for local development only. These are NOT the seeds you
# are graded on. Final scoring runs a PRIVATE, held-out seed set (same rules:
# 6 pieces, 2 distractors, no rotation; possibly different textures / cut shapes /
# layouts). Tuning to these sample seeds will not help your score, and an agent
# that only works on them will fail grading.
SAMPLE_SEEDS: List[int] = [101, 202, 303, 404, 505]


@dataclass
class ChallengeSpec:
    settings: Dict[str, object] = field(default_factory=lambda: dict(PUZZLE_SETTINGS))
    n_pieces: int = N_PIECES
    n_distractors: int = N_DISTRACTORS
    budget: int = DEFAULT_BUDGET
    #: Defaults to the public sample seeds. The grader overrides this with the
    #: private held-out set (``evaluate.py --seeds-file`` or ``$GRADER_SEEDS_FILE``).
    seeds: List[int] = field(default_factory=lambda: list(SAMPLE_SEEDS))

    def task_card(self) -> Dict[str, object]:
        """Public description handed to the agent at episode start."""
        return {
            "goal": ("Identify the distractor fragments. The board shows a picture "
                     "with a hole cut out of it (the silhouette is visible). The tray "
                     "holds fragments. Exactly N_DISTRACTORS of them belong to a "
                     "different (decoy) cut-out and fit nowhere in the hole. Return "
                     "their piece ids."),
            "n_pieces": self.n_pieces,
            "n_distractors": self.n_distractors,
            "rotation_enabled": False,
            "budget": self.budget,
            "submit_with": "tools.submit_distractors([id, id])",
        }
