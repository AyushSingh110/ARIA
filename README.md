# ARIA — Autonomous Reflective Intelligence Architecture

**An agent reliability system that detects, classifies, and explains why AI agents fail.**

ARIA ingests any agent trace (LangGraph, OpenAI, or raw tool calls), runs it through a multi-agent diagnostic pipeline, and produces a structured failure report — which requirements were missed, what behavioral failure occurred, and what to fix.

**Status:** ARIA Alpha — runtime API + dashboard live · validated on real-world and GAIA benchmark traces · paper targeting ICLR 2026 workshop / ACL 2026 System Demonstrations.

---

## What ARIA produces

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

---

## Headline results

| Metric | Value | Source |
|---|---|---|
| Failure-detection precision (flag → agent actually wrong) | **91.7%** | GAIA Level 1, 41 runs |
| Human–ARIA agreement on failure class | **68%** | 50 human-labeled real runs |
| Agreement improvement from Critic redesign | **8% → 42% → 68%** | Critic v1 → v2 → v2+rules |
| Diagnostician F1 on synthetic benchmark | **76.6%** overall | ARIA-Bench, 1,200 runs |

*Aggregate numbers only — raw traces, labeled datasets, and full analyses are private research assets.*

---

## Key research findings

### 1 — A 5-class behavioral failure taxonomy for tool-using agents
Validated on 1,200 synthetic runs (ARIA-Bench) and 90+ real traces:

| Class | Layer | Description |
|---|---|---|
| `prompt_drift` | Mechanism | Trajectory diverges from the original goal over turns |
| `tool_misuse` | Mechanism | Wrong tool, wrong args, or tool errors |
| `context_overflow` | Mechanism | Repeats completed steps; context lost |
| `hallucination_loop` | Mechanism | Asserts facts without tool grounding |
| `goal_misalignment` | **Outcome** | Task "completed" but requirements not satisfied |

**Insight:** `goal_misalignment` is an *outcome-level* label while the others are *mechanisms* — this structural difference explains its persistent classification overlap and motivates a two-layer taxonomy (v2).

### 2 — Holistic evaluation systematically overestimates agent success
A holistic output-quality Critic gave 5/5 correctness to runs that never produced required files, formulas, or sources. Redesigning the Critic around **per-requirement verification** (extract requirements → verify each against the trace → `requirement_satisfaction = satisfied/total`) raised human agreement from **8% to 42%**, and deterministic disambiguation rules raised it further to **68%**.

### 3 — The hallucination blind spot
Requirement-aware evaluation verifies that requirements are *addressed* — not that answers are *factually correct*. On GAIA, 3 of 4 false-clean cases had `requirement_satisfaction = 1.0` despite confidently wrong answers. This motivated **Critic v3: independent factual grounding** — extract the agent's central claim, verify it against independent web evidence, and flag contradicted claims as hallucinations.

### 4 — Failure-signal asymmetry on real traces
When ARIA flags a failure, the agent was actually wrong **91.7%** of the time (high precision). Real-world failure distributions differ sharply from synthetic ones: `goal_misalignment` dominates real traces (requirement omission, partial completion), while loop-type failures need larger turn budgets to manifest.

### 5 — Taxonomy gaps discovered through real usage
Human review surfaced failure modes outside the taxonomy ("gap" labels): agents returning stale or wrong information from a *successful* tool call — neither tool misuse nor hallucination. Sub-types documented: `partial_completion` (50%), `requirement_omission` (30%), `superficial_success` (10%).

---

## Architecture

```
Input: agent trace (any framework) or live task
          │
   Orchestrator ── decomposes task into subtasks
          │
     Executor ──── runs tools (web_search · calculator · read/write_file)
          │
     Observer ──── behavioral anomaly signals (drift, repetition, error loops)
          │
    Critic v2 ──── requirement extraction + per-requirement verification
          │
    Critic v3 ──── factual grounding of the final claim (hallucination check)
          │
  Diagnostician ── DSPy program + XGBoost + deterministic disambiguation rules
          │
   Refiner / Validator ── prompt component rewriting + scoring
          │
Output: structured failure report
```

**Stack:** LangGraph (cyclic StateGraph) · Groq Llama-3.3-70B · Ollama (local executor/critic) · DSPy BootstrapFewShot · XGBoost · SentenceTransformers · DuckDuckGo search · FastAPI · React + Recharts

---

## Quick start

### 1. Backend (Runtime API)

```bash
cd backend
pip install -r requirements.txt
cp .env.example .env          # add GROQ_API_KEY

uvicorn api.main:app --reload --port 8000
# Swagger docs: http://localhost:8000/docs
```

### 2. Dashboard

```bash
cd frontend
npm install
npm run dev
# Open: http://localhost:5173
```

### 3. Diagnose a trace in three lines

```python
from aria.sdk import diagnose

report = diagnose(
    task="Find the population of France and save it to population.txt",
    tool_calls=[{"tool_name": "web_search",
                 "tool_args": {"query": "population of France"},
                 "tool_result": "67.8 million (2024 estimate)..."}],
    final_output="The population of France is 67.8 million.",
)
print(report["failure_class"], report["requirement_satisfaction"])
```

### 4. Framework adapters

```python
# LangGraph
from adapters.langgraph_adapter import diagnose_langgraph_trace
report = diagnose_langgraph_trace(messages, task_description="...")

# OpenAI (Assistants run steps or chat completions with tool_calls)
from adapters.openai_adapter import diagnose_openai_trace
report = diagnose_openai_trace(run_steps, task_description="...")
```

### 5. Run a full pipeline task from the CLI

```bash
cd backend
python main.py "Calculate compound interest on $10,000 at 5% for 3 years and save it to results.txt"
```

---

## API endpoints

| Endpoint | Method | Purpose |
|---|---|---|
| `/diagnose` | POST | Diagnose a pre-computed trace |
| `/diagnose/batch` | POST | Diagnose multiple traces |
| `/run` | POST | Run a task through the full pipeline |
| `/feedback` | POST | Submit human correction (research-through-usage) |
| `/dashboard` | GET | Aggregate stats + distribution |
| `/health` | GET | Liveness |

Every diagnosis is saved as a potential training record; `/feedback` corrections become labeled data for Diagnostician recompilation — the system improves through usage.

---

## Project structure

```
backend/
  aria/
    agents/        orchestrator · executor · observer · critic (v2) · grounding (v3) ·
                   diagnostician · refiner · validator
    classifiers/   XGBoost failure classifier
    dspy_programs/ DSPy Diagnostician (5-field signature)
    graph/         LangGraph StateGraph builder
    sdk.py         3-line integration API
    state/ config/ memory/ store/ tools/ utils/
  adapters/        langgraph_adapter · openai_adapter
  api/             FastAPI runtime (main.py · schemas.py)
  scripts/         benchmark runners · labeling · analysis · DSPy compilation
  pyproject.toml   installable package: aria-agent-diagnostics
frontend/          React + Recharts dashboard (Vite)
research/          private — analyses, figures notebook (aggregates published in README)
```

---

## Roadmap

| Phase | Status |
|---|---|
| Research Cycle 0 — taxonomy + ARIA-Bench + DSPy Diagnostician | ✅ |
| Research Cycles 0.5/0.75 — confusion audit, ontology finding | ✅ |
| Research Cycle 1 — real-world validation (50 tasks, human-labeled) | ✅ |
| Critic v2 — requirement-aware evaluation | ✅ |
| ARIA Alpha — runtime API, batch diagnosis, feedback loop | ✅ |
| GAIA benchmark integration (Level 1) | ✅ |
| Critic v3 — factual grounding (hallucination blind spot) | ✅ |
| Framework adapters (LangGraph, OpenAI) + SDK | ✅ |
| Dashboard (React) | ✅ |
| Diagnostician v2 recompile on human-labeled data | 🔄 |
| GAIA Level 2 + ablation study | Planned |
| Taxonomy v2 — mechanism/outcome two-layer | Planned |
| Persistent experience store (DB-backed) | Planned |
| Paper submission | Targeting ICLR 2026 WS / ACL 2026 Demo |

---

## Research agenda

ARIA is stage two of a research line on AI reliability:

- **FIE** — *Did* failure occur? (output-level hallucination detection)
- **ARIA** — *Why* did failure occur? (behavioral failure diagnosis)
- **SAGE** (planned) — Can failure be *predicted* before execution?

Core claims: (1) an empirically validated behavioral failure taxonomy for tool-using agents; (2) goal misalignment is an outcome-level, not mechanism-level, label; (3) holistic evaluation overestimates agent success — requirement-aware evaluation is necessary; (4) requirement-aware evaluation alone misses confident factual errors — independent grounding closes the gap.

---

*Raw datasets, labeled traces, and full research analyses are not included in this repository.*
