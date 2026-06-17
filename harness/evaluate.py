"""Run an agent across the held-out puzzle set and score it.

Examples
--------
    # browserless smoke test (no Chromium needed)
    python -m harness.evaluate --agent probe --mock

    # the real test against headless Chromium
    python -m harness.evaluate --agent baselines.probe_agent:ProbeAgent

    # your own agent
    python -m harness.evaluate --agent my_pkg.my_agent:MyAgent --seeds 101 202 303

Writes results/<name>.json, results/<name>.trace.jsonl, and prints a summary table.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import pathlib
import statistics
import time
from typing import List, Optional

from . import make_bridge
from .agent_base import DistractorAgent
from .challenge import ChallengeSpec
from .session import BudgetExceeded, PuzzleSession

RESULTS = pathlib.Path(__file__).resolve().parents[1] / "results"


def load_agent(ref: str) -> DistractorAgent:
    """Resolve 'random' | 'probe' | 'example' or a 'module.path:ClassName' ref."""
    aliases = {
        "random": "baselines.random_agent:RandomAgent",
        "probe": "baselines.probe_agent:ProbeAgent",
        "example": "examples.example_agent:ExampleAgent",
    }
    ref = aliases.get(ref, ref)
    if ":" not in ref:
        raise SystemExit(f"--agent must be one of {list(aliases)} or 'module:Class', got {ref!r}")
    mod_name, cls_name = ref.split(":", 1)
    cls = getattr(importlib.import_module(mod_name), cls_name)
    return cls


def score_episode(predicted: List[int], truth: List[int]):
    pset, tset = set(predicted or []), set(truth)
    tp = len(pset & tset)
    fp = len(pset - tset)
    fn = len(tset - pset)
    return {
        "exact": pset == tset,
        "tp": tp, "fp": fp, "fn": fn,
        "precision": tp / (tp + fp) if (tp + fp) else 0.0,
        "recall": tp / (tp + fn) if (tp + fn) else 0.0,
    }


def run(agent_cls, spec: ChallengeSpec, seeds: List[int], mock: bool,
        vision: bool, headless: bool) -> dict:
    stateful = getattr(agent_cls, "STATEFUL", False)
    agent = agent_cls() if stateful else None
    bridge = make_bridge(mock=mock, headless=headless)
    bridge.start()
    episodes = []
    trace = []   # uniform session history across all episodes
    try:
        for seed in seeds:
            ep_agent = agent if stateful else agent_cls()
            sess = PuzzleSession(bridge, spec, seed, vision=vision)
            truth = sess._ground_truth_distractors()
            # Candidate agents get the locked proxy; only the oracle sanity
            # baseline (grader-only) is trusted with the full session.
            handed = sess if getattr(agent_cls, "NEEDS_ORACLE", False) else sess.tools()
            t0 = time.time()
            error = None
            ret = None
            try:
                ret = ep_agent.solve(handed)
            except BudgetExceeded as e:
                error = f"budget_exceeded: {e}"
            except Exception as e:  # candidate bug — recorded, not fatal
                error = f"{type(e).__name__}: {e}"
            wall = time.time() - t0
            if sess.submission is None and ret is not None and error is None:
                try:
                    sess.submit_distractors(ret)
                except Exception as e:
                    error = f"submit_error: {type(e).__name__}: {e}"
            trace.extend(sess.trace)
            predicted = sess.submission if sess.submission is not None else (ret or [])
            sc = score_episode(predicted, truth)
            episodes.append({
                "seed": seed, "predicted": sorted(predicted), "truth": truth,
                "calls_used": sess.calls_used, "over_budget": sess.over_budget,
                "wall_sec": round(wall, 3), "error": error, **sc,
            })
            print(f"seed={seed:<5} pred={sorted(predicted)} truth={truth} "
                  f"exact={sc['exact']} calls={sess.calls_used} {wall:.2f}s"
                  + (f"  ERROR {error}" if error else ""))
    finally:
        bridge.close()

    n = len(episodes)
    walls = [e["wall_sec"] for e in episodes]
    summary = {
        "agent": getattr(agent_cls, "name", agent_cls.__name__),
        "stateful": stateful, "mock": mock, "vision": vision, "n_episodes": n,
        "accuracy": round(sum(e["exact"] for e in episodes) / n, 4) if n else 0.0,
        "mean_precision": round(statistics.mean(e["precision"] for e in episodes), 4) if n else 0,
        "mean_recall": round(statistics.mean(e["recall"] for e in episodes), 4) if n else 0,
        "mean_calls": round(statistics.mean(e["calls_used"] for e in episodes), 2) if n else 0,
        "over_budget_rate": round(sum(e["over_budget"] for e in episodes) / n, 4) if n else 0,
        "error_rate": round(sum(bool(e["error"]) for e in episodes) / n, 4) if n else 0,
        "mean_wall_sec": round(statistics.mean(walls), 3) if walls else 0,
        "p95_wall_sec": round(sorted(walls)[int(0.95 * (n - 1))], 3) if walls else 0,
        "episodes": episodes,
        "_trace": trace,   # popped by main() and written to <name>.trace.jsonl
    }
    return summary


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", default="probe",
                    help="random|probe|example or 'module.path:ClassName'")
    ap.add_argument("--seeds", type=int, nargs="*", default=None,
                    help="run these explicit seeds")
    ap.add_argument("--seeds-file", default=None,
                    help="JSON file with a list of seeds (grader's private set); "
                         "falls back to $GRADER_SEEDS_FILE")
    ap.add_argument("--mock", action="store_true",
                    help="use the browserless mock bridge (CI / no Chromium)")
    ap.add_argument("--oracle-vision", action="store_true",
                    help="grader debug mode; only NEEDS_ORACLE agents receive the full session")
    ap.add_argument("--show", action="store_true", help="run Chromium headed (debug)")
    ap.add_argument("--budget", type=int, default=None, help="override per-episode budget")
    args = ap.parse_args(argv)

    spec = ChallengeSpec()
    if args.budget is not None:
        spec.budget = args.budget

    seeds_file = args.seeds_file or os.environ.get("GRADER_SEEDS_FILE")
    if args.seeds:
        seeds = args.seeds
    elif seeds_file:
        seeds = json.loads(pathlib.Path(seeds_file).read_text())
        print(f"[grader] loaded {len(seeds)} private seeds from {seeds_file}")
    else:
        seeds = spec.seeds
        print(f"[sample] using {len(seeds)} PUBLIC SAMPLE seeds — not the grading set")
    agent_cls = load_agent(args.agent)
    summary = run(agent_cls, spec, seeds, mock=args.mock,
                  vision=not args.oracle_vision, headless=not args.show)

    RESULTS.mkdir(exist_ok=True)
    trace = summary.pop("_trace", [])
    out = RESULTS / f"{summary['agent']}.json"
    out.write_text(json.dumps(summary, indent=2))
    trace_out = RESULTS / f"{summary['agent']}.trace.jsonl"
    with trace_out.open("w") as fh:
        for rec in trace:
            fh.write(json.dumps(rec) + "\n")
    print("\n" + "=" * 56)
    for k in ("agent", "n_episodes", "accuracy", "mean_precision", "mean_recall",
              "mean_calls", "over_budget_rate", "error_rate", "mean_wall_sec", "p95_wall_sec"):
        print(f"  {k:<18} {summary[k]}")
    print(f"  results -> {out}")
    print(f"  trace   -> {trace_out}  ({len(trace)} tool calls)")
    print("=" * 56)


if __name__ == "__main__":
    main()
