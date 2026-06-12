<div align="center">

# ARIA

**Diagnose *why* AI agents fail — not just *that* they failed.**

[![PyPI](https://img.shields.io/badge/pip%20install-ariadx-blue)](https://pypi.org/project/ariadx/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-green.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Status: Alpha](https://img.shields.io/badge/status-alpha-orange.svg)](#roadmap)

[Getting started](docs/getting-started.md) · [Documentation](docs/README.md) · [Research findings](#key-research-findings) · [Contributing](CONTRIBUTING.md)

</div>

---

ARIA (Autonomous Reflective Intelligence Architecture) ingests any agent trace — LangGraph, OpenAI, or raw tool calls — runs it through a multi-agent diagnostic pipeline, and produces a structured failure report: **which requirements were missed, what behavioral failure occurred, and what to fix.**

```python
from aria.sdk import diagnose

report = diagnose(
    task="Find the population of France and save it to population.txt",
    tool_calls=[{"tool_name": "web_search",
                 "tool_args": {"query": "population of France"},
                 "tool_result": "67.8 million (2024 estimate)..."}],
    final_output="The population of France is 67.8 million.",
)

report["failure_class"]              # "goal_misalignment" — file was never saved
report["requirements_satisfied"]     # [True, False]
report["suggested_action"]           # what to fix
```

## Why ARIA

Agent evaluation today mostly answers a binary question: did the run succeed? When it didn't, you're left reading traces by hand. ARIA answers the question that actually saves you time: **how did it fail?** — backed by a behavioral failure taxonomy validated against human judgment on real-world and benchmark traces.

## Headline results

| Metric | Value | Source |
|---|---|---|
| Failure-detection precision (when ARIA flags, agent was actually wrong) | **91.2%** | GAIA Level 1 benchmark |
| Human–ARIA agreement on failure class | **78%** | 50 human-labeled real-world runs |
| Agreement improvement through Critic redesign | **8% → 42% → 78%** | Critic v1 → v2 → v2 + rules + recompile |
| Diagnostician F1 on synthetic benchmark | **76.6%** | ARIA-Bench, 1,200 runs |

*Aggregate numbers only — raw traces, labeled datasets, and full analyses are private research assets.*

## Installation

```bash
pip install ariadx              # SDK + pipeline
pip install "ariadx[api]"       # + REST API server
```

Configure one LLM provider (free Groq key works for everything):

```bash
cp backend/.env.example .env    # add GROQ_API_KEY=gsk_...
```

**→ Full setup guide: [docs/getting-started.md](docs/getting-started.md)**

## Quick start

### Diagnose a trace (3 lines)

```python
from aria.sdk import diagnose
report = diagnose(task="...", tool_calls=[...], final_output="...")
print(report["failure_class"], report["requirement_satisfaction"])
```

### From LangGraph or OpenAI

```python
from adapters.langgraph_adapter import diagnose_langgraph_trace
report = diagnose_langgraph_trace(messages, task_description="...")

from adapters.openai_adapter import diagnose_openai_trace
report = diagnose_openai_trace(run_steps, task_description="...")
```

### REST API + dashboard

```bash
# Terminal 1 — API (Swagger at :8000/docs)
cd backend && uvicorn api.main:app --port 8000

# Terminal 2 — dashboard at :5173
cd frontend && npm install && npm run dev
```

### CLI — run a task through the full pipeline

```bash
aria "Calculate compound interest on $10,000 at 5% for 3 years and save to results.txt"
```

## The failure taxonomy

| Class | Layer | What it means |
|---|---|---|
| `prompt_drift` | mechanism | Trajectory diverges from the original goal over turns |
| `tool_misuse` | mechanism | Wrong tool, wrong args, or tool errors (requires actual error evidence) |
| `context_overflow` | mechanism | Repeats completed steps; context lost |
| `hallucination_loop` | mechanism | Asserts facts without grounding; contradicted by independent search |
| `goal_misalignment` | **outcome** | Task "completed" but requirements not satisfied |

**→ Full taxonomy with detection details: [docs/failure-taxonomy.md](docs/failure-taxonomy.md)**

## Architecture

![ARIA Diagnostic Pipeline](docs/architecture.png)

The pipeline is a cyclic LangGraph of seven agents. The three diagnostic innovations:

1. **Critic v2 — requirement-aware evaluation.** Extracts requirements from the task and verifies each one against the trace, instead of asking an LLM for a holistic quality score (which overestimates success).
2. **Critic v3 — independent factual grounding.** For clean-looking runs, extracts the agent's central claim and verifies it against an *independent* web search — catching confidently wrong answers that requirement checks miss.
3. **Diagnostician — LLM + rules hybrid.** A DSPy program compiled on human-labeled real traces, constrained by deterministic disambiguation rules that encode findings from human review.

**→ Deep-dive: [docs/architecture.md](docs/architecture.md)**

**Stack:** LangGraph · Groq Llama-3.3-70B · Ollama · DSPy · XGBoost · SentenceTransformers · FastAPI · React + Recharts

## Key research findings

1. **A validated 5-class behavioral failure taxonomy** for tool-using agents (1,200 synthetic + 90 real traces). `goal_misalignment` is an *outcome*-level label while the others are *mechanisms* — this structural difference explains its persistent classification overlap and motivates a two-layer taxonomy (v2, in design).

2. **Holistic evaluation systematically overestimates agent success.** A holistic LLM critic gave 5/5 correctness to runs that never produced required files. Redesigning evaluation around per-requirement verification raised human agreement from 8% to 42%; deterministic rules and recompilation on human labels raised it to 78%.

3. **The hallucination blind spot.** Requirement-aware evaluation verifies requirements are *addressed*, not *factually correct*. On GAIA, 3 of 4 false-clean cases had a perfect requirement score with a confidently wrong answer — motivating Critic v3's independent grounding.

4. **Failure-signal asymmetry on real traces.** ARIA's failure flags have 91.2% precision, but real-world failure distributions differ sharply from synthetic ones: `goal_misalignment` dominates real traces; loop-type failures need larger turn budgets to manifest.

5. **Taxonomy gaps discovered through usage.** Human review surfaced a failure mode outside the taxonomy: agents faithfully reporting *stale* information from a *successful* tool call — neither tool misuse nor hallucination. This drives taxonomy v2.

## Research-through-usage

Every diagnosis is saved as a candidate training record. Human corrections via `/feedback` (or the dashboard) become labeled data, and the Diagnostician is periodically recompiled on the accumulated labels — the deployed system and the research dataset improve together.

## Project structure

```
backend/
  aria/            # the 7 diagnostic agents, pipeline, SDK, CLI
  adapters/        # LangGraph + OpenAI adapters
  api/             # FastAPI runtime
  scripts/         # benchmarks, labeling, analysis (not shipped)
  pyproject.toml   # installable package: ariadx
frontend/          # React + Recharts dashboard
docs/              # user documentation
```

## Roadmap

| Milestone | Status |
|---|---|
| Failure taxonomy + ARIA-Bench (1,200 synthetic runs) + DSPy Diagnostician | ✅ |
| Real-world validation — 50 tasks, human-labeled | ✅ |
| Critic v2 — requirement-aware evaluation | ✅ |
| Runtime API + feedback loop + dashboard | ✅ |
| GAIA Level 1 benchmark | ✅ |
| Critic v3 — independent factual grounding | ✅ |
| Diagnostician v2 — recompiled on human-labeled real data | ✅ |
| SDK + framework adapters + PyPI package | ✅ |
| GAIA Level 2 + ablation study | 🔄 |
| Taxonomy v2 — mechanism/outcome two-layer | Planned |
| More adapters (CrewAI, AutoGen, smolagents) | Help wanted! |
| Paper | Targeting ICLR 2026 WS / ACL 2026 Demo |

## Research agenda

ARIA is stage two of a research line on AI reliability:

- **FIE** — *Did* failure occur? (output-level hallucination detection)
- **ARIA** — *Why* did failure occur? (behavioral failure diagnosis)
- **SAGE** (planned) — Can failure be *predicted* before execution?

## Contributing

Contributions are welcome — especially framework adapters, bug reports with traces, and research discussion on the taxonomy. Start with [CONTRIBUTING.md](CONTRIBUTING.md).

## License

[Apache 2.0](LICENSE)

---

*Raw datasets, labeled traces, and full research analyses are not included in this repository.*
