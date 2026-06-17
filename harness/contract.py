"""Observation/Action contract shared by the env and any agent.

Mirrors window.SpatialGame. Two observation modes:
  - oracle : full geometry (cut-out, vertices, home targets, metrics)
  - vision : rendered pixels + minimal addressable handles (no targets)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Action:
    type: str               # 'move' | 'rotate' | 'place'
    id: int
    x: Optional[float] = None
    y: Optional[float] = None
    rot: Optional[float] = None
    cost: Optional[float] = None

    def to_js(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"type": self.type, "id": self.id}
        for k in ("x", "y", "rot", "cost"):
            v = getattr(self, k)
            if v is not None:
                d[k] = v
        return d


@dataclass
class Observation:
    mode: str                                  # 'oracle' | 'vision'
    solved: bool
    failed: bool
    placed_count: int
    pieces: List[Dict[str, Any]] = field(default_factory=list)
    constraints: Dict[str, Any] = field(default_factory=dict)
    canvas: Dict[str, Any] = field(default_factory=dict)
    image_png_b64: Optional[str] = None        # vision mode only (data URL stripped)
    cutout: Optional[List[Dict[str, float]]] = None   # oracle only
    metrics: Optional[Dict[str, Any]] = None          # oracle only
    raw: Dict[str, Any] = field(default_factory=dict)

    @property
    def unplaced(self) -> List[Dict[str, Any]]:
        return [p for p in self.pieces if not p.get("placed")]
