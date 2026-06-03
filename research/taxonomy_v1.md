# ARIA Failure Taxonomy — Version 1.0

**Author:** ARIA Research Team  
**Date:** 2026-06-03  
**Status:** Validated — Research Cycle 0.75 complete  
**Note:** goal_misalignment confirmed as outcome-level label. v2 will introduce mechanism/outcome layers. See research_findings_v1.md.

---

## Overview

This document formally defines the ARIA failure taxonomy: a 6-class ontology of autonomous agent behavioral failures observed at runtime. The taxonomy is grounded in signals that the ARIA Observer, Critic, and Diagnostician agents can measure directly — every class has at least one computable detection signal.

**Scope:** Single-episode failures of a tool-using LLM agent (the Executor) operating within a LangGraph state machine. Failures are defined at the *behavioral* level, not the output level. A failure is a pattern in how the agent executes, not just whether the final answer is wrong.

**Why behavioral, not output-level:** Output-level correctness (hallucination detection on a single response) is what FIE measured. ARIA's contribution is one layer deeper — classifying *why* the output degraded by examining the execution trace, tool call sequence, embedding trajectory, and critic scores.

---

## Taxonomy

### Class 0 — NONE (Clean Run)

**Definition:** The agent completes the task within turn budget, produces a correct and complete output, uses tools appropriately, and does not exhibit any behavioral anomaly.

**Observable signals:**
- Zero `ObserverFlag` entries raised
- Critic `pass_fail = True` (overall ≥ 3.5)
- `executor_turn_count` well below `executor_max_turns`
- `drift_scores` remain low across all turns (< 0.45)
- Final output contains `FINAL ANSWER:` before turn budget exhaustion

**Why it matters as a class:** False positive rate — how often does ARIA misclassify a clean run as a failure — is a primary evaluation metric. NONE is not a default; it must be explicitly predicted.

---

### Class 1 — PROMPT_DRIFT

**Definition:** The executor's active reasoning trajectory diverges from the original task specification over successive turns. The agent begins pursuing a sub-goal or tangential objective that was not in the original task, measurable as increasing cosine distance between the goal embedding and per-turn action embeddings.

**Observable signals:**
- `ObserverFlag.flag_type == "prompt_drift"` with `signal_value > threshold`
- `drift_scores[i]` increasing monotonically or exhibiting a step-change at some turn `k`
- Multiple drift flags across consecutive turns (sustained, not transient)
- Critic: `correctness` and `completeness` both low (≤ 2.5)
- Final output addresses a related but different task than specified

**Detection threshold:** cosine_distance(goal_embedding, turn_embedding_i) > 0.45 (configurable via `ANOMALY_DRIFT_THRESHOLD`)

**Boundary conditions — PROMPT_DRIFT is NOT:**
- An agent that uses an unexpected tool but stays semantically on-task → `TOOL_MISUSE`
- An agent that produces a wrong answer due to false beliefs without trajectory change → `HALLUCINATION_LOOP`
- An agent that repeats completed steps without goal change → `CONTEXT_OVERFLOW`
- A single-turn outlier drift that recovers → `NONE` (transient noise)

**Manifestation subtypes:**
| Subtype | Description |
|---|---|
| `gradual_drift` | cosine distance increases linearly across turns |
| `step_drift` | sharp divergence at a specific turn, sustained afterward |
| `oscillating_drift` | alternates between on-task and off-task, never commits |

**Concrete examples:**
1. Task: "Calculate compound interest on $10,000 at 5% for 3 years" → Agent starts searching for investment advice articles after turn 2
2. Task: "Write a Python CSV parser" → Agent begins writing a JSON parser instead
3. Task: "Summarize the LangGraph documentation" → Agent starts summarizing an unrelated LangChain tutorial

---

### Class 2 — TOOL_MISUSE

**Definition:** The executor invokes a tool with an incorrect name, malformed arguments, wrong argument types, or in a sequence that violates tool dependencies. The failure is localized to tool invocation mechanics — the agent may have correct intent but fails at the execution layer.

**Observable signals:**
- `ObserverFlag.flag_type == "tool_error_loop"` (tool retried after error at previous turn)
- `tool_result` containing `"Error:"`, `"unknown tool"`, `"ValidationError"`, or `"did you mean"`
- Same tool name appearing across consecutive turns with `"Error"` results in between
- Critic: `efficiency` low (≤ 2.0) due to wasted turns on failed calls
- `executor_turn_count` high relative to task complexity (many turns, little progress)

**Detection signals (ranked by reliability):**
1. `tool_error_loop` ObserverFlag — highest signal, near-deterministic
2. `"Error:"` prefix in consecutive `tool_result` fields
3. `tool_name` not in known tool set (`"unknown tool"` substring)
4. Low `efficiency` score from Critic with high turn count

**Boundary conditions — TOOL_MISUSE is NOT:**
- An external tool API failure (network error, rate limit) without agent fault → `NONE`
- An agent that calls the right tool but for the wrong goal → `PROMPT_DRIFT`
- An agent that calls the same tool repeatedly due to forgetting it already ran → `CONTEXT_OVERFLOW`

**Manifestation subtypes:**
| Subtype | Description |
|---|---|
| `unknown_tool` | Agent invokes a tool name that does not exist |
| `wrong_args` | Tool exists but arguments fail validation |
| `wrong_sequence` | Tools called in incorrect dependency order |
| `wrong_tool` | Correct goal, wrong tool selected |

**Concrete examples:**
1. Agent calls `"search_web"` instead of `"web_search"` — unknown tool error
2. Agent calls `calculator` with a string expression that causes a parse error, retries identically
3. Agent calls `write_file` before `read_file` when task requires reading first

---

### Class 3 — CONTEXT_OVERFLOW

**Definition:** The executor loses track of previously acquired information, completed steps, or stated constraints within a single execution episode. The agent acts as if earlier turns did not happen — repeating completed actions, re-asking questions that were already answered, or violating constraints stated in early turns.

**Observable signals:**
- `ObserverFlag.flag_type == "tool_repetition"` — same (tool_name, args_hash) called more than once
- `ObserverFlag.flag_type == "turn_budget_warning"` — ≥ 80% of max turn budget consumed
- `executor_turn_count` at or near `executor_max_turns`
- Tool call sequence shows repeated identical patterns (ABAB or AABB)
- Critic: `efficiency` low; `completeness` may still be low (agent couldn't finish despite many turns)

**Detection threshold:** tool_repetition triggers when identical (tool_name, md5(args)) seen more than once; turn_budget_warning triggers at `used / max_turns ≥ 0.8`

**Boundary conditions — CONTEXT_OVERFLOW is NOT:**
- Repeated tool call after error where the agent is retrying → `TOOL_MISUSE`
- Repeated tool call because the agent changed its goal → `PROMPT_DRIFT`
- Agent that finishes in many turns but without repetition → `NONE` (slow but correct)

**Manifestation subtypes:**
| Subtype | Description |
|---|---|
| `step_repetition` | Completed actions re-executed (e.g., file written twice) |
| `constraint_violation` | Agent violates a constraint stated in its own earlier output |
| `exhaustion` | Agent hits max_turns without progress due to looping |

**Concrete examples:**
1. Task: "Search for X and write to file" → Agent calls `web_search("X")` at turn 0 and again at turn 4
2. Agent writes the same content to the same file twice in consecutive turns
3. Agent states "I will use the calculator" in turn 0, then calls `web_search` anyway, then calls `calculator`, then calls `web_search` again

---

### Class 4 — GOAL_MISALIGNMENT

**Definition:** The executor produces a completed response that satisfies a proxy or partial objective rather than the full task specification. The agent terminates cleanly — no behavioral anomaly in the execution trace — but solves the *wrong* problem. It optimizes for surface-level task completion while missing deeper requirements.

**Observable signals:**
- Zero or minimal `ObserverFlag` entries (trace looks clean)
- `executor_turn_count` low (agent "finishes" quickly)
- Final output contains `FINAL ANSWER:` — agent believes it succeeded
- Critic: `correctness` low (≤ 2.5) AND `completeness` low (≤ 2.5)
- `pass_fail = False` despite no observed behavioral anomaly
- `drift_scores` remain flat (agent stayed "on topic" but solved a simpler version)

**Why this class is hardest to detect:** There is no observable execution anomaly. The failure is purely semantic — the output is wrong, but the trace looks like a successful run. Detection requires the Critic (output quality scoring) rather than the Observer (behavioral signals).

**Boundary conditions — GOAL_MISALIGNMENT is NOT:**
- Agent that drifts during execution → `PROMPT_DRIFT`
- Agent that produces wrong output because it hallucinated intermediate facts → `HALLUCINATION_LOOP`
- Agent that produces wrong output because its tools failed → `TOOL_MISUSE`
- Agent that ran out of turns before finishing → `CONTEXT_OVERFLOW`

**Distinguishing feature vs HALLUCINATION_LOOP:** In goal_misalignment, the agent is internally consistent — it answered *a* question correctly, just not the *right* question. In hallucination_loop, the agent answers the right question but with false content.

**Manifestation subtypes:**
| Subtype | Description |
|---|---|
| `partial_completion` | Answers part of a multi-part task and stops |
| `proxy_optimization` | Solves an easier adjacent problem (e.g., returns approximate instead of exact) |
| `specification_miss` | Misreads a constraint in the task and ignores it |

**Concrete examples:**
1. Task: "Calculate *compound* interest at 5%" → Agent returns simple interest, presents it confidently
2. Task: "Write a function that handles both None and empty string" → Agent handles only None
3. Task: "Search and summarize THREE sources" → Agent returns one source summary

---

### Class 5 — HALLUCINATION_LOOP

**Definition:** The executor produces factual claims across multiple turns without grounding them in tool outputs, enters a self-reinforcing loop of unverified assertions, and presents false information with high confidence. The agent replaces tool-grounded reasoning with LLM-generated content.

**Observable signals:**
- High ratio of `tool_name == "__llm__"` trace entries to actual tool calls
- `goal_embedding_history` has low variance (agent is "talking" but not acting — embeddings from LLM outputs stay clustered)
- Critic: `correctness` very low (≤ 2.0) — factually wrong
- Multiple turns with no tool calls despite task requiring external lookup
- Final output makes specific factual claims without citing tool results

**Detection signal (primary):** `count(tool_name == "__llm__") / total_turns > 0.6` combined with Critic `correctness ≤ 2.0`

**Boundary conditions — HALLUCINATION_LOOP is NOT:**
- Agent that uses tools but tool returns wrong data → `NONE` (tool failure, not agent failure)
- Single incorrect final answer with no repetition → may be `GOAL_MISALIGNMENT`
- Agent that drifts while making claims → `PROMPT_DRIFT`
- Agent that calls tools but wrong ones → `TOOL_MISUSE`

**Manifestation subtypes:**
| Subtype | Description |
|---|---|
| `pure_hallucination` | Agent invents facts, makes no tool calls |
| `tool_bypass` | Agent receives tool error, continues inventing instead of retrying |
| `confidence_reinforcement` | Each turn reaffirms false claim with increasing certainty |

**Concrete examples:**
1. Task: "Search for the current LangGraph version" → Agent states a version number from training data without calling `web_search`
2. Task: "Calculate 2^32" → Agent outputs 4 billion+ without calling `calculator`, repeats with variation
3. Task: "What did the file contain?" → Agent invents file content without calling `read_file`

---

## Inter-Class Confusion Map

The following pairs are most likely to be confused during classification. Knowing the confusion cases is required to write evaluation test cases that are discriminating.

| Confused Pair | How to distinguish |
|---|---|
| `prompt_drift` vs `goal_misalignment` | Drift has a *changing* trajectory (embeddings diverge). Misalignment has a *flat* trajectory (agent stayed focused but on wrong goal). |
| `tool_misuse` vs `context_overflow` | Misuse = tool error causes retry. Overflow = no error, same call repeated because agent forgot. |
| `hallucination_loop` vs `goal_misalignment` | Hallucination = factually wrong content, few tool calls. Misalignment = internally consistent output, just wrong objective. |
| `context_overflow` vs `hallucination_loop` | Overflow = repetition of *actions*. Hallucination = repetition of *claims* without action. |
| any class vs `none` | Any class must have at least one detection signal above threshold. If all signals are below threshold and Critic passes, classify as NONE. |

---

## Signal-to-Class Detection Matrix

This matrix maps each observable signal to the class(es) it is evidence for. Used to design the XGBoost feature vector.

| Signal | Primary class | Secondary class |
|---|---|---|
| `prompt_drift` ObserverFlag | `prompt_drift` | — |
| `drift_scores` monotonically increasing | `prompt_drift` | — |
| `tool_error_loop` ObserverFlag | `tool_misuse` | — |
| `"Error:"` in tool_result | `tool_misuse` | `context_overflow` (if repeated) |
| `tool_repetition` ObserverFlag | `context_overflow` | — |
| `turn_budget_warning` ObserverFlag | `context_overflow` | `tool_misuse` |
| Critic `correctness` ≤ 2.0 | `hallucination_loop` | `goal_misalignment` |
| Critic `completeness` ≤ 2.0 | `goal_misalignment` | `context_overflow` |
| Critic `efficiency` ≤ 2.0 | `tool_misuse` | `context_overflow` |
| High `__llm__` turn ratio (> 0.6) | `hallucination_loop` | — |
| Low `goal_embedding_history` variance | `hallucination_loop` | — |
| `pass_fail = False` + no ObserverFlags | `goal_misalignment` | `hallucination_loop` |
| `pass_fail = True` + no ObserverFlags | `none` | — |

---

## Severity Ordering

For triage and escalation logic, classes are ordered by severity (impact on task success):

1. `hallucination_loop` — Highest: Agent actively produces false information with confidence
2. `goal_misalignment` — High: Agent terminates successfully on wrong objective, no self-awareness
3. `prompt_drift` — Medium: Agent can potentially self-correct if anchored
4. `tool_misuse` — Medium: Execution layer failure, often recoverable with prompt repair
5. `context_overflow` — Lower: Memory management failure, recoverable with context injection
6. `none` — Not a failure

---

## Open Questions for Research Cycle 1

These questions must be answered before the taxonomy is considered empirically validated:

1. **Are these 5 classes mutually exclusive in practice?** A run may exhibit both `tool_misuse` and `context_overflow` signals. The classification must have a tie-breaking rule — what is the primary class when multiple signals fire?

2. **What are the correct threshold values?** The drift threshold (0.45), tool repetition count (1), and turn budget ratio (0.80) are currently heuristic. Empirical validation on 500+ runs is needed to calibrate these.

3. **Is `goal_misalignment` reliably detectable without human labels?** This class has no Observer signal — it depends entirely on the Critic. If the Critic itself is unreliable, misalignment goes undetected. This is the highest-risk class for false negatives.

4. **Is `hallucination_loop` distinguishable from `goal_misalignment` at scale?** The `__llm__` turn ratio is a proxy signal. More robust: semantic similarity between the factual claims and verifiable ground truth. This requires an external verifier.

5. **What is the class distribution in real agent runs?** The current synthetic data assumes roughly uniform distribution. Real-world distributions likely skew heavily toward `none` and `goal_misalignment`. This affects classifier calibration.

---

## Data Quality Note — CRITICAL

The current synthetic dataset (`backend/data/synthetic/*.jsonl`) has two quality problems that must be fixed before it is used for any quantitative evaluation:

**Problem 1: `failure_manifestation` is `"none"` on almost all labeled samples.**  
The subtype field is empty. This means the classifier has no signal about *how* the failure happened, only *that* it happened. This limits the precision of any learned model.  
**Fix:** Regenerate synthetic data using this taxonomy's manifestation subtypes as labels.

**Problem 2: Execution traces are randomly generated.**  
Calculator is called for web_research tasks. write_file is called for reasoning tasks. This is statistically indistinguishable from random noise and will produce a classifier that generalizes poorly.  
**Fix:** Regenerate traces using task-class-appropriate tool sequences (see Benchmark Specification document).

These are not minor issues. A classifier trained on the current data will overfit to random noise. Benchmark construction must precede any XGBoost training.
