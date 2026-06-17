# Agent specification — what to demonstrate

This is a take-home about **agent design**, not puzzle-solving trivia. The puzzle
is deliberately easy to state so your effort goes into the agent. Below is what we
look for; you don't need heavy machinery to show each one, but your design should
make a deliberate choice about each.

## Required capabilities

**Tool calling.** Wrap the `tools.*` methods (a locked `ToolsProxy`) as
first-class tools in your framework (ADK / LangGraph / Deep Agents) and let the
agent invoke them. We want to see the model choosing tools, not a hard-coded
script that happens to import a framework. Keep the wiring minimal — we grade the
pattern, not framework ceremony.

**Context invocation & governance.** Be deliberate about what goes into the model
context and when. Don't dump every piece image on every step. Respect the
per-episode budget (default 60 tool-units) — treat it as a real resource. Failing
gracefully when near the limit is part of the test.

**Memory management.** Track what you've learned within an episode (which pieces
look real, which probes you've tried) so you don't re-pay for the same
observation. If your agent is `STATEFUL`, you may also carry knowledge *across*
episodes — show us you thought about what's worth remembering and what isn't.

**Stateful vs. stateless execution.** Declare `STATEFUL` on your agent and justify
it in NOTES.md. Stateless (fresh per puzzle) is simpler and often the right call;
stateful lets you amortize learning across puzzles. Either is fine — the reasoning
is what we grade.

**Execution primitives (loop / goal).** Structure the run as an explicit loop
toward a goal ("I am confident about both distractors") with a clear stopping
condition, rather than a fixed number of steps. Show the control flow.

## Strategy is open

There are several legitimate ways to win, and your choice signals your taste:

- **Perception:** read the board and fragment crops; reason about which shapes
  could tile the visible hole. Cheap in tool calls, leans on the model.
- **Probing:** use `try_fit` feedback as ground truth (real pieces snap,
  distractors never do). Robust but expensive — watch the budget.
- **Hybrid:** perception to rank candidates, a few probes to confirm. Usually the
  sweet spot.

The `probe` baseline shows brute-force probing *works* but costs ~190 calls and
dies under a tight budget. Beating it on **efficiency**, not just accuracy, is
where strong candidates separate.

## How we score (5 dimensions)

| # | Dimension | What we're judging |
|---|-----------|--------------------|
| 1 | **Agent design & the harness you build** | the agent (tool calling, control loop, memory, governance) *and* the harness around it: how you wrap the tools, your **prompt & instruction design**, orchestration, guardrails, and a readable trace |
| 2 | **Execution quality** | accuracy on the held-out seeds (did it actually find the distractors) |
| 3 | **Runtime & performance** | tool-call cost, latency, behaviour under the budget |
| 4 | **Code quality** | readability, structure, tests, error handling, reproducibility |
| 5 | **Taste — judgement & originality** | the right idea for the constraints with trade-offs reasoned out; elegant restraint over theatrical cleverness. Judged largely from your **prompts, instructions, and session history** |

We run `python -m harness.evaluate --agent <yours>` on a held-out seed set and read
your `results/<name>.json`, **your prompts/instructions, your session history/trace**,
and `NOTES.md` alongside the code. Aim for an agent that is **accurate, cheap, and
easy to read** — in that order. Submit the artifacts listed under "Submitting" in
the README; missing prompts/instructions or trace will hold back dimensions 1 and 5.
