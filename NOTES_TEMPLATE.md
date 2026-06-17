# Candidate Notes Template

Use this as the shape for your `NOTES.md`. Keep it to one page.

## Strategy

What approach did you choose: perception, probing, or hybrid? Why this strategy
for a 60-call budget, and what did you deliberately reject?

## Agent Harness

Which framework did you use, and how are the `tools.*` methods wrapped as tools?
Where does the model make decisions versus fixed control flow?

## Context And Budget Policy

What goes into the model context at each step? What do you avoid sending? How do
you reserve budget and decide when to stop?

## Memory And State

Is `STATEFUL` true or false? What do you remember within an episode, and what
would be worth carrying across episodes?

## Results

Include your command, `results/<name>.json`, and `results/<name>.trace.jsonl`.
Summarize accuracy, mean calls, latency, and any over-budget/error episodes.

## Failure Mode

Name one case where your agent can fail and how you would improve it with more
time.

## AI Tool Usage

Which parts did you use AI coding tools for, and how did you verify or change
their output?
