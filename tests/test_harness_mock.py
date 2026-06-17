"""Browserless tests over the deterministic MockBridge.

Run:  pytest -q     (no Chromium required)

These verify the harness contract candidates rely on:
  * puzzles have exactly 4 real + 2 distractor pieces, no rotation;
  * the answer key is hidden in vision mode;
  * try_fit snaps real pieces and never snaps distractors;
  * the budget is enforced;
  * the evaluator scores the floor / heuristic / ceiling agents sensibly.
"""
from __future__ import annotations

import sys
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from harness import make_bridge
from harness.agent_base import DistractorAgent
from harness.challenge import ChallengeSpec
from harness.session import BudgetExceeded, PuzzleSession
from harness.evaluate import load_agent, run


def _session(seed=101, vision=True, budget=None):
    spec = ChallengeSpec()
    if budget is not None:
        spec.budget = budget
    bridge = make_bridge(mock=True)
    bridge.start()
    return PuzzleSession(bridge, spec, seed, vision=vision), bridge, spec


def test_puzzle_composition():
    sess, bridge, spec = _session()
    truth = sess._ground_truth_distractors()
    pieces = sess.list_pieces()
    assert len(pieces) == spec.n_pieces == 6
    assert len(truth) == spec.n_distractors == 2
    # no rotation anywhere
    for p in bridge.get_state(vision=False)["pieces"]:
        assert p["rot"] == 0.0
    bridge.close()


def test_answer_key_hidden_in_vision():
    sess, bridge, _ = _session(vision=True)
    vstate = bridge.get_state(vision=True)
    for p in vstate["pieces"]:
        assert "isDistractor" not in p and "home" not in p
    bridge.close()


def test_try_fit_discriminates():
    sess, bridge, _ = _session()
    truth = set(sess._ground_truth_distractors())
    oracle = {p["id"]: p for p in bridge.get_state(vision=False)["pieces"]}
    for pid, p in oracle.items():
        res = sess.try_fit(pid, p["home"]["x"], p["home"]["y"])
        if pid in truth:
            assert res["snapped"] is False     # distractors never snap
        else:
            assert res["snapped"] is True      # real piece snaps at its home
    bridge.close()


def test_budget_enforced():
    sess, bridge, _ = _session(budget=5)
    raised = False
    try:
        for _ in range(100):
            sess.list_pieces()
    except BudgetExceeded:
        raised = True
    assert raised and sess.over_budget
    bridge.close()


def test_proxy_blocks_answer_key_access():
    sess, bridge, _ = _session()
    tools = sess.tools()
    # sanctioned tools work
    assert len(tools.list_pieces()) == 6
    assert isinstance(tools.budget, int) and tools.calls_used >= 0
    # everything that could leak the answer is unreachable
    for forbidden in ("bridge", "spec", "seed", "_ground_truth_distractors",
                      "_obs", "__dict__", "_ToolsProxy__s"):
        assert not hasattr(tools, forbidden), f"proxy leaks {forbidden}"
    # cannot stash a back-reference either (slots)
    import pytest
    with pytest.raises(AttributeError):
        tools.bridge = bridge
    with pytest.raises(AttributeError):
        getattr(tools, "_ToolsProxy__s")
    assert "_ToolsProxy__s" not in dir(tools)
    bridge.close()


def test_trace_records_tool_calls():
    sess, bridge, _ = _session()
    sess.list_pieces()
    sess.try_fit(0, 100, 100)
    sess.submit_distractors([0, 1])
    tools_seen = [r["tool"] for r in sess.trace]
    assert "list_pieces" in tools_seen
    assert "try_fit" in tools_seen
    assert "submit_distractors" in tools_seen
    # every record has the uniform schema and no raw image bytes
    for r in sess.trace:
        assert set(("seq", "t", "seed", "tool", "args", "result", "cost", "calls_used")) <= r.keys()
        assert r["seed"] == 101
    # image tools summarise to a byte count, never raw bytes
    sess.board_image_png()
    img_rec = [r for r in sess.trace if r["tool"] == "board_image_png"][-1]
    assert "bytes" in img_rec["result"] and isinstance(img_rec["result"]["bytes"], int)
    bridge.close()


def test_returned_answer_is_submitted_and_traced():
    class ReturnOnlyAgent(DistractorAgent):
        name = "return-only"

        def solve(self, tools):
            tools.task()
            return [0, 1]

    spec = ChallengeSpec()
    summary = run(ReturnOnlyAgent, spec, [101], mock=True, vision=True, headless=True)
    assert summary["episodes"][0]["predicted"] == [0, 1]
    assert summary["_trace"][-1]["tool"] == "submit_distractors"
    assert summary["_trace"][-1]["args"] == {"ids": [0, 1]}


def test_oracle_agent_is_perfect():
    spec = ChallengeSpec()
    summary = run(load_agent("baselines.oracle_agent:OracleAgent"), spec, spec.seeds,
                  mock=True, vision=False, headless=True)
    assert summary["accuracy"] == 1.0


def test_random_agent_is_weak():
    spec = ChallengeSpec()
    summary = run(load_agent("random"), spec, spec.seeds, mock=True, vision=True, headless=True)
    assert summary["accuracy"] < 0.5  # ~1/15 in expectation


def test_probe_agent_solves_with_budget():
    spec = ChallengeSpec()
    spec.budget = 400  # generous: brute probing is expensive but works
    summary = run(load_agent("probe"), spec, spec.seeds, mock=True, vision=True, headless=True)
    assert summary["accuracy"] >= 0.8
