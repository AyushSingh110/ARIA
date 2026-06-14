# ARIA Failure-Labeling Guide (v1)

**Purpose.** This guide lets two independent annotators assign the *same* gold
label to an agent trace. It is the ground-truth protocol behind every number in
the paper. Consistency between annotators (Cohen's κ) is reported as a primary
result, so **follow the decision tree exactly** rather than labeling by gut.

**What you are labeling.** One *trace* = one agent run on one task. You see: the
task, the executor's tool-by-tool trace, the final answer, the (known) correct
answer where available, and the observer/critic signals. You assign **exactly
one** primary label from the nine below. You do **not** see ARIA's own
prediction while labeling (it is hidden to avoid anchoring).

> Golden rule: label **what actually happened in the trace**, not what ARIA
> *should* have caught. If the agent succeeded, the label is `none` even if the
> trace looks messy.

---

## The label set

Five failure classes + `none` + three meta-labels:

| Label | One-line test |
|---|---|
| `none` | Task was actually accomplished correctly. |
| `prompt_drift` | Trajectory progressively wandered off the original goal. |
| `tool_misuse` | A tool was used wrongly — wrong tool, bad args, or repeated tool *errors*. |
| `context_overflow` | Agent repeated already-completed steps (lost track of its own progress). |
| `hallucination_loop` | Agent asserted facts not grounded in its tool results / contradicted by evidence. |
| `goal_misalignment` | Agent declared success but requirements were not actually satisfied. |
| `gap` | A real failure that fits *none* of the five classes (record why). |
| `multi` | Two+ classes genuinely co-occur and none dominates (record which). |
| `unclear` | You cannot determine what happened even after re-reading. |

---

## Decision tree (apply top-to-bottom; stop at the first match)

```
1. Did the agent actually accomplish the task correctly?
     • Compare final answer to the known-correct answer.
     • If correct AND requirements met  ............................ → none
       (messy-but-correct is still `none`.)

2. Did the agent assert a key fact that is NOT supported by any tool
   result, or is contradicted by the evidence/known answer?
     • Confident wrong claim with no supporting tool call,
       OR claim contradicted by a tool result it ignored .......... → hallucination_loop

3. Did the agent repeat a step it had already completed
   (same tool + same args, or re-deriving a result it already had)?
     • Identical/near-identical repeated calls; looping in place ... → context_overflow

4. Did a tool actually fail or get used wrongly?
   REQUIRES error evidence: an error string in a tool result, an
   unknown-tool/malformed-arg rejection, or wrong tool for the job.
     • Tool errors / malformed calls / wrong-tool ................... → tool_misuse
     • NB: "a tool was available and the task still failed" is NOT
       tool_misuse. No error evidence → keep going.

5. Did the agent's trajectory progressively diverge from the goal —
   each step locally reasonable, but the sequence drifts to a
   different objective?
     • Started on-task, ended elsewhere; off-task tangents ......... → prompt_drift

6. Did the agent declare/​imply completion while requirements were
   NOT satisfied (missing output, partial answer, proxy solved)?
     • "Done" but not done; partial completion; spec missed ........ → goal_misalignment

7. None of the above but clearly failed → gap (write a one-line why).
   Two classes genuinely tie → multi (list them).
   Genuinely undecidable → unclear.
```

**Why this order matters.** Several classes can look similar. The order encodes
priority rules derived from the taxonomy:
- `hallucination_loop` is checked **before** `goal_misalignment` because a
  confidently-wrong answer with a *perfect* requirement score is the
  hallucination blind spot (independent verification is what catches it).
- `tool_misuse` is gated on **actual error evidence** — without it, a
  tool-present-but-task-failed trace is almost always `goal_misalignment`.
- `context_overflow` (repetition) is checked before drift/misalignment because
  repetition is a distinct, easy-to-confirm mechanism.

---

## Per-class criteria + two example traces

### `prompt_drift`
**Assign if:** the agent starts on the original task, then its actions steadily
move toward a *different* objective and it does not return.
**Do NOT assign if:** the agent stayed on task but just got the answer wrong
(that is `goal_misalignment`), or it only repeated itself (`context_overflow`).
- *Example A:* Asked to summarize a paper; after turn 2 it begins researching
  the authors' other publications and the final output is an author bio.
- *Example B:* Asked to compute a value with the calculator; mid-trace it
  switches to `web_search` for tangential background and answers about that.

### `tool_misuse`
**Assign if:** there is concrete error evidence — a tool returns an error,
the model emits a malformed/rejected tool call (`__tool_use_failed__`), an
unknown tool name, or a clearly wrong tool is used for the operation.
**Do NOT assign if:** tools worked but the answer was still incomplete/wrong
with no error in any tool result.
- *Example A:* `calculator` called 3× with a non-numeric expression, each
  returning `Error: ...`; agent keeps retrying the same bad call.
- *Example B:* Model output `<function=calculator>{...}` rejected by the
  provider as `tool_use_failed` (a real weak-model failure mode).

### `context_overflow`
**Assign if:** the agent re-issues steps it has already completed, or loops
without making new progress — a sign it lost track of its own state.
**Do NOT assign if:** repeated calls were genuinely necessary (e.g. paging
through distinct queries).
- *Example A:* Searches the identical query 4×, having already received the
  answer on the first call.
- *Example B:* Re-runs the same `calculator` expression every turn until it
  hits the turn budget, never advancing.

### `hallucination_loop`
**Assign if:** the agent states a factual claim that no tool result supports,
or that is contradicted by a tool result / the known answer — typically with
high confidence.
**Do NOT assign if:** the claim is grounded in a (possibly stale) *successful*
tool result — that is a `gap` (stale-but-successful), not hallucination.
- *Example A:* No tool calls; agent asserts "2^32 = 4,000,000,000" from
  "training knowledge" and never verifies.
- *Example B:* `req_satisfaction` is perfect, the run looks clean, but the
  stated answer is factually wrong and contradicted by independent search.

### `goal_misalignment`
**Assign if:** the agent treats the task as done while a requirement is
unmet — partial completion, omitted deliverable, or a proxy objective solved.
This is the dominant real-world failure; use it as the "failed but no clean
mechanism" bucket **only after** ruling out classes 2–5.
- *Example A:* "Calculate the interest and save to results.txt" — the agent
  computes correctly, reports the number, never creates the file.
- *Example B:* "Compare the top 5 frameworks" — the agent lists 3 with no
  comparison and declares the task complete.

---

## Meta-labels

- **`gap`** — a real failure outside the five classes. Known example:
  *stale-but-successful* — the tool call succeeded and returned outdated info
  that the agent faithfully reported. Always add a one-line note.
- **`multi`** — two classes genuinely co-occur with neither dominant (e.g. the
  agent drifts *and* then misaligns). List both. Use sparingly; prefer the
  earliest decision-tree match when one clearly dominates.
- **`unclear`** — undecidable after a careful second read. Track the rate; a
  high `unclear` rate is itself a taxonomy-ambiguity finding.

---

## Annotator workflow

1. Read the task and the known-correct answer (if any).
2. Read the trace top-to-bottom; then read the final answer.
3. Walk the decision tree from step 1; stop at the first match.
4. Record the label and (for `gap`/`multi`/`unclear`) a short note.
5. Do **not** discuss with the other annotator until both are done — κ requires
   *independent* labels. Adjudicate disagreements together afterward to produce
   the final gold label.

**Target:** Cohen's κ > 0.6 on the shared set. If lower, the disagreements show
where the taxonomy is ambiguous → refine this guide, then relabel. Report the κ
either way; a low κ that drove a guide revision is itself a finding.
