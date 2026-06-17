"""Playwright bridge to the game's ``window.SpatialGame`` seam.

Drives the real HTML/JS engine in headless Chromium so the agent reasons over the
exact pixels and geometry a human would see.

Requires (run once locally):
    pip install -r requirements.txt
    playwright install chromium
"""
from __future__ import annotations

import base64
import pathlib
from typing import Any, Dict, Optional

# repo_root/game/index.html  (browser.py lives in repo_root/harness/)
_DEFAULT_GAME = pathlib.Path(__file__).resolve().parents[1] / "game" / "index.html"


class GameBridge:
    """Thin wrapper over the ``window.SpatialGame`` JS API."""

    def __init__(self, game_path: Optional[str] = None, url: Optional[str] = None,
                 headless: bool = True):
        self._game_path = game_path or str(_DEFAULT_GAME)
        self._url = url
        self._headless = headless
        self._pw = None
        self._browser = None
        self.page = None

    def __enter__(self) -> "GameBridge":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def start(self) -> None:
        from playwright.sync_api import sync_playwright  # lazy: tests can run without it
        self._pw = sync_playwright().start()
        self._browser = self._pw.chromium.launch(headless=self._headless)
        self.page = self._browser.new_page(viewport={"width": 900, "height": 820})
        target = self._url or pathlib.Path(self._game_path).as_uri()
        self.page.goto(target)
        self.page.wait_for_function("() => !!window.SpatialGame")

    def close(self) -> None:
        try:
            if self._browser:
                self._browser.close()
        finally:
            if self._pw:
                self._pw.stop()

    # --- seam wrappers ---
    def set_settings(self, settings: Dict[str, Any]) -> Dict[str, Any]:
        return self.page.evaluate("(s) => window.SpatialGame.setSettings(s)", settings)

    def set_seed(self, seed) -> Dict[str, Any]:
        return self.page.evaluate("(s) => window.SpatialGame.setSeed(s)", seed)

    def new_game(self) -> Dict[str, Any]:
        return self.page.evaluate("() => window.SpatialGame.newGame()")

    def get_state(self, vision: bool = False) -> Dict[str, Any]:
        return self.page.evaluate("(v) => window.SpatialGame.getState({vision:v})", vision)

    def get_observation(self) -> Dict[str, Any]:
        return self.page.evaluate("() => window.SpatialGame.getObservation()")

    def apply_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        return self.page.evaluate("(a) => window.SpatialGame.applyAction(a)", action)

    @staticmethod
    def dataurl_to_b64(data_url: str) -> str:
        if data_url and "," in data_url:
            return data_url.split(",", 1)[1]
        return data_url or ""

    @staticmethod
    def b64_to_bytes(b64: str) -> bytes:
        return base64.b64decode(b64) if b64 else b""
