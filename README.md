# Spatial Distractor-Detection Challenge

A take-home for engineers who build agents. You will build an **agent** that looks
at a spatial puzzle and decides which pieces *don't belong*.

**Time budget: ~2 hours.** We are testing your judgement and agent-design taste,
not your stamina. A clean, well-reasoned partial solution beats a sprawling one.

---

## The puzzle

A picture has a shape cut out of it — the hole's silhouette is visible on the
board. The cut-out was shattered into **4 fragments**, which sit scattered in the
tray below. Mixed in with them are **2 distractor fragments** that were cut from a
*different* decoy shape and **fit nowhere** in the hole.

So the tray holds **6 pieces; exactly 2 are distractors.** There are **no
rotations** — pieces only translate.

```
┌──────────────────────────────────────┐
│  picture with a hole cut out          │
│        ╭───────╮      (silhouette)    │
│        │  hole │                      │
│        ╰───────╯                      │
├──────────────────────────────────────┤
│  tray:  [0] [1] [2] [3] [4] [5]       │   ← 4 fit the hole, 2 are decoys
└──────────────────────────────────────┘
```

## Your task

Build an agent that identifies the **2 distractor piece ids**. That's the whole
job — you don't have to solve the puzzle, just spot the impostors.

```python
tools.submit_distractors([id_a, id_b])
```

You decide *how*: reason over the rendered pixels, probe placements and learn from
feedback, or combine both. There are several valid strategies, and we care about
the one you pick and why.

## What you must use

Build a real agent with **one** of:

- [Google ADK](https://google.github.io/adk-docs/)
- [LangGraph](https://langchain-ai.github.io/langgraph/)
- [Deep Agents](https://github.com/langchain-ai/deepagents)

The harness is framework-agnostic — wrap the `tools.*` methods (below) as tools in
your framework and let your agent drive. **We grade the agent pattern, not
framework ceremony:** a minimal, clean graph/loop is perfectly fine — don't add
nodes for show. **Using AI coding tools is allowed and expected** — just tell us
how you used them in `NOTES.md`. See **[AGENT_SPEC.md](AGENT_SPEC.md)** for the
capabilities we want exercised and how we grade.

---

## The harness

Your agent is handed a locked `ToolsProxy` (`harness/session.py`) — a small,
**metered** tool surface and nothing else. Every call costs against a per-episode
budget (default **60**); blow the budget and the episode ends. There is no
supported path through `tools` to the browser, settings, seed, or answer key.
Reaching around the proxy or using private/oracle fields by any means is an
automatic fail (we run static checks and read traces). Solve it honestly; that's
the whole point.

| Tool | What it does | Cost |
|------|--------------|------|
| `task()` | goal, piece/distractor counts, budget, board dimensions | free |
| `list_pieces()` | `[{id, x, y, placed}, …]` — addressable handles | 1 |
| `board_image_png()` | PNG of the whole board (hole silhouette + tray) | 1 |
| `piece_image_png(id)` | a square crop zoomed on one fragment | 1 |
| `try_fit(id, x, y)` | drop a piece at (x,y); returns `{snapped}` — a real piece can snap home, a distractor never will | 1 |
| `remember(k,v)` / `recall(k)` | scratch memory (or use your framework's) | free |
| `submit_distractors([ids])` | final answer; ends the episode | free |

Implement the interface in `harness/agent_base.py`:

```python
from harness.agent_base import DistractorAgent

class MyAgent(DistractorAgent):
    name = "my-agent"
    STATEFUL = False          # True keeps one instance across all puzzles

    def solve(self, tools) -> list[int]:
        ...                   # inspect via tools, then:
        tools.submit_distractors([a, b])
        return [a, b]
```

## Quick start

```bash
pip install -r requirements.txt
playwright install chromium          # one-time: downloads headless Chromium

# 1) browserless sanity check (no Chromium needed)
pytest -q
python -m harness.evaluate --agent example --mock

# 2) the real thing against the actual game
python -m harness.evaluate --agent example
python -m harness.evaluate --agent probe --budget 400    # heuristic baseline

# 3) your agent
python -m harness.evaluate --agent your_pkg.your_module:YourAgent
```

`evaluate.py` runs your agent across a set of seeded puzzles and writes
`results/<name>.json` (accuracy, tool-call cost, latency, budget/error rates) and
`results/<name>.trace.jsonl` — a uniform, one-line-per-tool-call **session
history** the harness records for you (tool, args, cost, running total). Submit
that file as-is. Reference baselines: `random` ≈ 5% (floor), `probe` = 100% but
~190 calls (works, but expensive — beat it on efficiency).

> **Seeds:** the seeds in `harness/challenge.py` are **public samples for dev
> only**. Final scoring runs a **private, held-out** seed set under the same rules
> (6 pieces, 2 distractors, no rotation; possibly different textures / cut shapes).
> An agent that only works on the sample seeds will fail. Build for generalization.

## Submitting

Ship via **GitHub** (public repo or invite us) **or Google Drive** (zipped). Include:

1. **Your agent code** + how to run it (one command).
2. **Your prompts and agent/system instructions** — the actual text you send the
   model (a `prompts/` file, or clearly inline). We read these closely.
3. **Session history / tool-call trace** — the harness writes
   `results/<name>.trace.jsonl` automatically; include it. Add your framework's
   own trace too if it's richer. You should be able to explain why you spent each
   class of call.
4. **`results/<name>.json`** from your own run.
5. A short **`NOTES.md`** (≤1 page; `NOTES_TEMPLATE.md` is provided) covering:
   - the strategy you chose and **why** (and what you rejected);
   - your **tool-budget policy** and **context policy** (what you put in the model
     context, and what you deliberately left out);
   - **stateful vs. stateless** choice and the reason;
   - a **failure mode** you observed or expect;
   - **how you used AI coding tools** (which parts, and how you checked them).
6. Don't commit secrets — read API keys from environment variables.

Heads-up: expect a **~15-minute debrief** where we'll ask you to walk a trace and
adapt the agent live (e.g. a smaller budget or an extra distractor).

Questions → reach out. Good luck.

---
MIT licensed. The game in `game/index.html` is provided as-is for this challenge.
