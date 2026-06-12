# SDK Reference

```python
from aria.sdk import diagnose, diagnose_remote, run_task
```

The SDK has three functions. `diagnose` is what most users need.

---

## `diagnose(task, tool_calls=None, final_output="")`

Diagnose a pre-computed agent trace **in-process** — no server required.

**Requirements:** `GROQ_API_KEY` set in the environment (or `.env`).

### Parameters

| Name | Type | Description |
|---|---|---|
| `task` | `str` | The task the agent was asked to perform |
| `tool_calls` | `list[dict]` | The tools the agent called, in order (see format below) |
| `final_output` | `str` | The agent's final answer text |

### `tool_calls` format

```python
[
    {
        "tool_name": "web_search",                       # required
        "tool_args": {"query": "population of France"},  # optional, default {}
        "tool_result": "67.8 million (2024)...",         # optional, default ""
        "turn": 0,                                       # optional, auto-assigned by position
    },
]
```

### Returns

```python
{
    "task_id": "d-20260612-...",
    "failure_class": "goal_misalignment",   # or None if clean
    "confidence": 0.84,
    "reasoning": "Requirement 'save to file' was never addressed...",
    "manifestation": "requirement omission",
    "suggested_action": "Add explicit success criteria...",

    # Requirement-aware evaluation (Critic v2)
    "requirement_satisfaction": 0.5,        # fraction of requirements met
    "requirements": ["find population", "save to population.txt"],
    "requirements_satisfied": [True, False],
    "evidence": ["Requirement not satisfied: 'save to population.txt'"],

    # Behavioral signals (Observer)
    "observer_flags": [{"flag_type": "tool_repetition", "turn": 3, ...}],
    "critic_scores": {"correctness": 4.0, "completeness": 2.0, ...},
    "executor_turn_count": 1,
    "trace_summary": "turn 0: web_search(...) -> 67.8 million...",
}
```

### Failure classes

| Value | Meaning |
|---|---|
| `None` | No failure detected — clean run |
| `"prompt_drift"` | Agent's trajectory diverged from the goal over turns |
| `"tool_misuse"` | Wrong tool, malformed args, or tool error loop |
| `"context_overflow"` | Agent repeats already-completed steps |
| `"hallucination_loop"` | Asserts facts without grounding / contradicted by evidence |
| `"goal_misalignment"` | Task "completed" but requirements not satisfied |

### Example

```python
from aria.sdk import diagnose

report = diagnose(
    task="Calculate compound interest on $10,000 at 5% for 3 years and save to results.txt",
    tool_calls=[
        {"tool_name": "calculator", "tool_args": {"expression": "10000 * 1.05**3"},
         "tool_result": "11576.25"},
    ],
    final_output="The compound interest result is $11,576.25.",
)

if report["failure_class"]:
    print(f"FAILED: {report['failure_class']} ({report['confidence']:.0%})")
    for req, ok in zip(report["requirements"], report["requirements_satisfied"]):
        print(f"  {'✓' if ok else '✗'} {req}")
```

---

## `diagnose_remote(task, tool_calls=None, final_output="", aria_url="http://localhost:8000", timeout=120.0)`

Same inputs and outputs as `diagnose`, but sends the trace to a **running ARIA API server** over HTTP. Use this when:

- You don't want LLM dependencies in your app process
- Multiple services share one ARIA instance
- You want every diagnosis recorded centrally (dashboard, feedback loop)

```python
from aria.sdk import diagnose_remote

report = diagnose_remote(
    task="...",
    tool_calls=[...],
    final_output="...",
    aria_url="http://aria.internal:8000",
)
```

Raises `httpx.HTTPStatusError` on non-2xx responses.

---

## `run_task(task, task_class="general")`

Run a task through ARIA's **full pipeline** — Orchestrator decomposes it, the Executor actually performs it with tools, and the diagnostic chain evaluates the result. This is the "self-driving" mode used for benchmarking.

**Requirements:** `GROQ_API_KEY`; Ollama if `EXECUTOR_PROVIDER=ollama` (default).

```python
from aria.sdk import run_task

result = run_task("Find the latest stable Python version and save it to version.txt")

result["output"]                    # what the agent produced
result["failure_class"]             # ARIA's diagnosis of its own run
result["requirement_satisfaction"]
result["grounding"]                 # Critic v3 verdict, if grounding triggered
```

### `task_class` values

`general` (default) · `code_generation` · `web_research` · `data_analysis` · `reasoning` · `tool_chaining`

---

## Error handling

| Situation | Behavior |
|---|---|
| `GROQ_API_KEY` missing | `RuntimeError` at first LLM call with a clear message |
| Groq rate limit (429) | Exception propagates — wrap in retry if running batches |
| Empty `tool_calls` | Valid — ARIA diagnoses from task + final output alone |
| Tool result very long | Automatically truncated before LLM calls |

---

## CLI

The package installs an `aria` command:

```bash
aria "Calculate 15% tip on $84.50 and save to tip.txt"
aria "Research LangGraph and summarize" --task-class web_research --verbose
```
