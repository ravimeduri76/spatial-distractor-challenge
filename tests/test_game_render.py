"""Real-browser rendering check (skipped when Chromium is unavailable).

The hole must be fully opaque: an agent may see the silhouette and the picture
*around* the hole, but never the picture *inside* it. A translucent fill leaks
each real piece's home position through template matching.

Run:  pytest -q tests/test_game_render.py   (needs `playwright install chromium`)
"""
from __future__ import annotations

import io
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from harness.challenge import ChallengeSpec

PIL = pytest.importorskip("PIL")
from PIL import Image, ImageDraw, ImageFilter  # noqa: E402

HOLE_FILL = (5, 8, 12)
EDGE_PX = 4          # skip the outline stroke and anti-aliased edge


@pytest.fixture(scope="module")
def bridge():
    pytest.importorskip("playwright")
    from harness.browser import GameBridge
    b = GameBridge()
    try:
        b.start()
    except Exception as e:                      # no Chromium installed
        pytest.skip(f"Chromium unavailable: {e}")
    yield b
    b.close()


@pytest.mark.parametrize("seed", ChallengeSpec().seeds)
def test_hole_interior_is_opaque(bridge, seed):
    spec = ChallengeSpec()
    bridge.set_settings({**spec.settings, "vision": False})
    bridge.set_seed(seed)
    bridge.new_game()
    # newGame() only marks the canvas dirty; wait for the redraw before reading pixels.
    bridge.page.evaluate("() => new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)))")
    state = bridge.get_state(vision=False)
    img = Image.open(io.BytesIO(bridge.b64_to_bytes(
        bridge.dataurl_to_b64(bridge.get_observation()["image"])))).convert("RGB")

    scale = img.width / state["canvas"]["w"]
    mask = Image.new("L", img.size, 0)
    ImageDraw.Draw(mask).polygon(
        [(p["x"] * scale, p["y"] * scale) for p in state["cutout"]], fill=255)
    mask = mask.filter(ImageFilter.MinFilter(2 * int(EDGE_PX * scale) + 1))

    inside = [px for px, m in zip(img.getdata(), mask.getdata()) if m]
    assert inside, "hole mask is empty"
    leaked = sum(px != HOLE_FILL for px in inside)
    assert leaked == 0, f"{leaked}/{len(inside)} hole pixels show the picture underneath"
