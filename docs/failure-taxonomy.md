# The ARIA Failure Taxonomy

ARIA classifies agent failures into **five behavioral classes**, validated on 1,200 synthetic runs (ARIA-Bench) and 90+ real-world traces with human labels.

---

## The five classes

### `prompt_drift` — the agent wanders off-goal

The agent's trajectory progressively diverges from the original task. Each step looks locally reasonable, but the sequence drifts.

- **Detected by:** embedding distance between the original goal and each turn's action (Observer drift signal, threshold 0.45 cosine distance)
- **Example:** asked to summarize a paper, the agent starts researching the authors' other work and never returns

### `tool_misuse` — the tools are used wrong

Wrong tool for the job, malformed arguments, or repeated tool errors.

- **Detected by:** `tool_error_loop` Observer flag, or error text in tool results
- **Important:** ARIA requires *actual error evidence* before assigning this class. Early versions over-predicted tool_misuse 3× whenever "tools were present and something went wrong" — a deterministic rule now corrects this (28% → 10% over-prediction eliminated on our benchmark)

### `context_overflow` — the agent loses track

The agent repeats steps it already completed, signaling lost context.

- **Detected by:** `tool_repetition` flags — identical tool+args called multiple times
- **Example:** searches for the same query 4 times, having forgotten it already has the answer

### `hallucination_loop` — confident fabrication

The agent asserts facts that have no grounding in its tool results, or that contradict independent evidence.

- **Detected by:** two mechanisms:
  1. Claims in the output with no supporting tool call (mechanism signal)
  2. **Critic v3 grounding** — ARIA extracts the agent's central factual claim, runs an *independent* web search, and flags claims contradicted by the evidence
- **Why two mechanisms:** requirement-aware evaluation alone has a blind spot — on the GAIA benchmark, 3 of 4 missed failures had a *perfect* requirement score with a confidently *wrong* answer. Only independent verification catches these.

### `goal_misalignment` — "done" but not done

The agent declares the task complete, but requirements were not satisfied. The dominant real-world failure (36% of our labeled real traces).

- **Detected by:** Critic v2 requirement extraction + per-requirement verification → `requirement_satisfaction < threshold`
- **Sub-types observed in human review:** partial completion (50%), requirement omission (30%), superficial success (10%)
- **Example:** "Calculate the interest and save to results.txt" — the agent calculates correctly, reports the number, and never creates the file

---

## A structural insight: mechanisms vs. outcomes

Four classes describe **mechanisms** — *how* the agent's behavior broke (drift, tool errors, repetition, ungrounded claims). `goal_misalignment` describes an **outcome** — *what* ended up wrong, regardless of mechanism.

This is not a flaw but a discovered property of the space: an agent can drift *and* misalign, or hallucinate *its way into* misalignment. The persistent classification overlap between `goal_misalignment` and the mechanism classes in both synthetic and real data motivated our planned **two-layer taxonomy (v2)**: a mechanism layer and an outcome layer assigned independently.

---

## What the taxonomy doesn't cover (yet)

Human review of real traces surfaced failures outside the five classes (labeled `gap` in our data, 4% of real runs):

- **Stale-but-successful tool results:** the agent gets a *successful* tool response containing outdated information and reports it faithfully. Not tool misuse (the call worked), not hallucination (the claim is grounded — in stale evidence).

These gap cases drive taxonomy v2 design.

---

## How classification actually happens

```
trace → Observer flags → Critic v2 requirement scores → Critic v3 grounding
                                   ↓
                  DSPy DiagnosticProgram (compiled on human-labeled data)
                                   ↓
                  Deterministic disambiguation rules (hard overrides)
                                   ↓
                          final failure_class
```

The deterministic rules encode findings from human-labeled data, e.g.:

- `req_sat = 0` can never be labeled "none"
- `tool_misuse` requires actual error evidence, else reclassify
- clean-looking runs (`req_sat ≥ 0.75`, no flags) with a grounding-contradicted claim become `hallucination_loop`

This LLM + rules hybrid raised human agreement from 8% (holistic LLM critic) to 42% (requirement-aware critic) to **78%** (with rules + recompiled diagnostician).
