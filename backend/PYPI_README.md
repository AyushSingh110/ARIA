# ariadx

**Diagnose *why* AI agents fail — not just *that* they failed.**

ARIA (Autonomous Reflective Intelligence Architecture) ingests any agent trace — LangGraph, OpenAI, or raw tool calls — and produces a structured failure report: which requirements were missed, what behavioral failure occurred, and what to fix.

```bash
pip install ariadx
```

## Three-line diagnosis

```python
from aria.sdk import diagnose

report = diagnose(
    task="Find the population of France and save it to population.txt",
    tool_calls=[{"tool_name": "web_search",
                 "tool_args": {"query": "population of France"},
                 "tool_result": "67.8 million (2024 estimate)..."}],
    final_output="The population of France is 67.8 million.",
)
print(report["failure_class"])               # e.g. "goal_misalignment"
print(report["requirement_satisfaction"])    # e.g. 0.5 — file was never saved
```

## What you get back

```json
{
  "failure_class": "goal_misalignment",
  "confidence": 0.84,
  "requirement_satisfaction": 0.43,
  "requirements": ["calculate compound interest", "show the formula", "save to results.txt"],
  "requirements_satisfied": [true, true, false],
  "evidence": ["Requirement not satisfied: 'save to results.txt'"],
  "suggested_action": "Add explicit success criteria and require the agent to verify them."
}
```

## Failure taxonomy

| Class | What it means |
|---|---|
| `prompt_drift` | Trajectory diverges from the original goal over turns |
| `tool_misuse` | Wrong tool, wrong args, or tool errors |
| `context_overflow` | Repeats completed steps; context lost |
| `hallucination_loop` | Asserts facts without grounding (caught by independent web verification) |
| `goal_misalignment` | Task "completed" but requirements not satisfied |

## Validated on real data

- **91%+ failure-detection precision** on the GAIA benchmark
- **78% human agreement** on failure classification (50 human-labeled real-world runs)
- Built on a 5-class behavioral failure taxonomy validated across 1,200 synthetic + 90 real agent traces

## Framework adapters

```python
from adapters.langgraph_adapter import diagnose_langgraph_trace
from adapters.openai_adapter import diagnose_openai_trace
```

## Runtime API + dashboard

```bash
pip install "ariadx[api]"
uvicorn api.main:app --port 8000     # REST API with /diagnose, /run, /feedback, /dashboard
```

## Links

- **Source & docs:** https://github.com/AyushSingh110/ARIA
- **License:** Apache 2.0
- **Research:** paper targeting ICLR 2026 workshop / ACL 2026 System Demonstrations
