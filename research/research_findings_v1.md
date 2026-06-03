# ARIA Research Findings — v1

**Date:** 2026-06-03  
**Stage:** Research Cycle 0.75 — Misclassification Analysis  
**Status:** Confirmed findings, documented for paper

---

## Finding 1 — Observable vs Silent Failure Split (Confirmed)

**Source:** DSPy validation run v1 (79.2% overall accuracy)

Three classes are behaviorally observable via the Observer agent:
- `context_overflow` — F1 100%, recall 100%
- `none` — F1 90.9%
- `prompt_drift` — F1 84.2%

Two classes are silent failures with zero Observer flags:
- `goal_misalignment` — F1 58.8%
- `hallucination_loop` — F1 57.1%, recall 40%

**Implication:** ARIA's Observer-based detection is reliable for mechanism-level failures. Silent failures require Critic-based semantic judgment, which is structurally harder.

---

## Finding 2 — Goal Misalignment is an Outcome, Not a Mechanism (Confirmed)

**Source:** Misclassification analysis + data fix experiment (Research Cycle 0.75)

**Evidence:**
- After fixing the data generator to produce task-specific goal_misalignment traces, F1 dropped further (58.8% to 50%) rather than improving.
- The confusion shifted from tool_misuse to prompt_drift — the target class changed but the confusion persisted.
- Two consecutive data fixes failed to resolve the ambiguity.

**Interpretation:**
The persistence of confusion after dataset correction indicates the issue is not data quality. It is ontology structure.

`goal_misalignment` describes what the user experienced (wrong objective achieved), not why the failure happened. This makes it an outcome-level label, while the other four classes are mechanism-level labels:

| Class | Type | Describes |
|---|---|---|
| `prompt_drift` | Mechanism | Why the failure happened |
| `tool_misuse` | Mechanism | Why the failure happened |
| `context_overflow` | Mechanism | Why the failure happened |
| `hallucination_loop` | Mechanism | Why the failure happened |
| `goal_misalignment` | **Outcome** | What the user experienced |

**Consequence:** The model correctly observes that goal_misalignment can be caused by multiple mechanisms (tool_misuse, prompt_drift, hallucination_loop). When forced to choose a single sibling label, it picks the most likely cause — which varies by example. This is not a classification error. It is a correct observation about a flawed taxonomy structure.

**Paper language (ready to use):**

> The persistence of goal_misalignment confusion after dataset correction suggests the issue is not data quality but ontology structure. Unlike the other categories, goal_misalignment functions as an outcome-level label rather than a mechanism-level label, creating unavoidable overlap with causal failure classes such as prompt_drift and tool_misuse. This finding motivates a two-layer taxonomy for ARIA v2: a mechanism layer (prompt_drift, tool_misuse, context_overflow, hallucination_loop) and an outcome layer (goal_misalignment, partial_completion, task_failure, recovery_failure).

---

## Research Risk Log

### Risk 1 — hallucination_loop vs goal_misalignment (Active, partially mitigated)
- **Status:** Hard pair confusion reduced from 2 to 0 after data fix.
- **Remaining risk:** hallucination_loop recall is 40% on 5 examples — too small a sample to conclude.
- **Action:** Do not build HallucinationBench yet. Keep v1 stable first.

### Risk 2 — goal_misalignment ontology overlap (Confirmed, documented)
- **Status:** Confirmed as structural ontology problem, not dataset problem.
- **Action:** Keep goal_misalignment in v1 taxonomy. Accept lower F1 as honest reflection of difficulty. Document for paper. Plan v2 two-layer taxonomy.

---

## ARIA v1 vs v2 Taxonomy Direction

### v1 (current) — flat 5-class taxonomy
All five classes treated as siblings. Goal_misalignment included as a first-class label. F1 ~50-58% on goal_misalignment is accepted as a known limitation.

### v2 (planned) — two-layer taxonomy

**Layer 1 — Failure Mechanisms** (observable, diagnosable)
- prompt_drift
- tool_misuse
- context_overflow
- hallucination_loop

**Layer 2 — Failure Outcomes** (what the user experiences)
- goal_misalignment
- partial_completion
- task_failure
- recovery_failure

Under v2, a single run can have one mechanism label AND one outcome label. The compound-interest example becomes: mechanism=tool_misuse, outcome=goal_misalignment. The ambiguity disappears because both labels are simultaneously correct.

---

## Next Steps (ARIA v1 completion)

1. Keep current DSPy model and v1 taxonomy as-is.
2. Run ARIA on real agent tasks (not synthetic) — first real-world validation.
3. Measure whether the observable classes (context_overflow, prompt_drift, tool_misuse) hold up on real traces.
4. Publish v1 results with honest limitations documented.
5. Begin v2 taxonomy design after v1 paper draft is complete.
