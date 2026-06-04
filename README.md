# ARIA — Autonomous Reflective Intelligence Architecture

An agent reliability research system that detects, classifies, and explains why AI agents fail.

**Current status:** Research validated. ARIA Alpha implementation in progress.

---

## What ARIA Does

ARIA runs an agent task through a 7-agent diagnostic pipeline and produces a structured failure report:

```json
{
  "failure_type": "goal_misalignment",
  "confidence": 0.84,
  "requirement_satisfaction": 0.43,
  "evidence": [
    "required file not created",
    "step 2 of 3 never executed"
  ],
  "suggested_action": "Add explicit success criteria the agent must verify before terminating."
}
```

---

## Research Context

ARIA is the second stage of a multi-project research agenda on AI Reliability Across Time:

- **FIE** — "Did failure occur?" (output-level hallucination detection)
- **ARIA** — "Why did failure occur?" (behavioral failure diagnosis)
- **SAGE** (planned) — "Can failure be predicted before execution?"

---

## Key Research Findings

### Finding 1 — Failure Taxonomy (Research Cycle 0)

ARIA defines a 5-class behavioral failure taxonomy validated on 1,200 synthetic runs (ARIA-Bench):

| Class | Type | Description |
|---|---|---|
| `prompt_drift` | Mechanism | Agent trajectory diverges from original goal over turns |
| `tool_misuse` | Mechanism | Wrong tool, wrong args, or wrong sequence |
| `context_overflow` | Mechanism | Agent repeats completed steps; context lost |
| `hallucination_loop` | Mechanism | Agent asserts facts without tool grounding |
| `goal_misalignment` | **Outcome** | Agent completes task but satisfies wrong objective |

**Key insight:** `goal_misalignment` behaves as an outcome-level label (what the user experienced), not a mechanism-level label (why it happened). This structural difference explains why it consistently overlaps with other classes in classification. ARIA v2 will introduce a two-layer taxonomy: mechanism layer + outcome layer.

DSPy Diagnostician accuracy on ARIA-Bench (held-out):
- `context_overflow`: 100% F1
- `none` (clean runs): 90.9% F1
- `prompt_drift`: 84.2% F1
- `tool_misuse`: 69.6% F1
- `goal_misalignment`: 50% F1 (structural — outcome label over mechanism labels)
- `hallucination_loop`: 57.1% F1

### Finding 2 — Holistic Evaluation Misses Requirement Failures (Research Cycle 1)

**The core finding:**

> Agent evaluation based on holistic output quality systematically overestimates success rates by failing to detect requirement omissions and partial task completion.

Running 50 real agent tasks through ARIA revealed:
- Critic v1 (holistic scoring) gave 5.0/5 correctness to runs where the agent never saved a required file, never computed a required formula, never found the required number of sources
- ARIA/human agreement with Critic v1: **8%**
- ARIA/human agreement with Critic v2 (requirement-aware): **42%**

Real-world failure distribution (50 tasks, human-labeled):
```
none (clean)        34%   ← mentor predicted 50-70%
goal_misalignment   48%   ← mentor predicted 15-25% (significantly higher)
tool_misuse         16%
prompt_drift         0%   (requires higher turn budget to manifest)
context_overflow     2%
hallucination_loop   0%   (requires higher turn budget to manifest)
```

**Gap failure sub-types discovered** (Research Cycle 1.25):
- `partial_completion` (50%) — agent completed step 1 of an N-step task
- `requirement_omission` (30%) — agent ignored an explicit stated constraint
- `superficial_success` (10%) — agent appeared done but used wrong/no source

### Finding 3 — Critic v2 Architecture

The Critic was redesigned from holistic output scoring to requirement-aware evaluation:

**v1 approach:** Score output quality holistically (correctness 1-5, completeness 1-5)
**v2 approach:**
1. Extract explicit requirements from task as a checklist
2. Verify each requirement against trace + output individually
3. `requirement_satisfaction = satisfied / total`
4. `pass_fail = requirement_satisfaction >= 0.75 AND correctness >= 2.0`

This single change produced the 5x improvement in human-ARIA agreement.

---

## Architecture

```
Input: Task Description
         ↓
    Orchestrator          — decomposes task into subtasks
         ↓
     Executor             — runs tool calls (calculator, web_search, write_file, read_file)
         ↓
     Observer             — detects behavioral anomalies (drift, repetition, errors)
         ↓
    Critic v2             — requirement-aware evaluation (not holistic scoring)
         ↓
  Diagnostician           — classifies failure class via DSPy + XGBoost
         ↓
     Refiner              — rewrites broken prompt component
         ↓
    Validator             — scores refinement quality
         ↓
Output: Failure Report
```

**Stack:** LangGraph (cyclic StateGraph), Groq (Llama-3.3-70B), Ollama (local), DSPy BootstrapFewShot, XGBoost, SentenceTransformers, FastAPI

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r backend/requirements.txt

# 2. Configure environment
cp backend/.env.example backend/.env
# Edit .env — add GROQ_API_KEY

# 3. Run a single task
cd backend
python main.py "Calculate compound interest on $10,000 at 5% for 3 years"

# 4. Start the ARIA Runtime API
uvicorn api.main:app --reload --port 8000
# Open: http://localhost:8000/docs
```

---

## Project Structure

```
backend/
  aria/
    agents/       orchestrator · executor · observer · critic · diagnostician · refiner · validator
    classifiers/  XGBoost failure classifier
    config/       Pydantic Settings (env-driven)
    dspy_programs/ DSPy Diagnostician (compiled)
    graph/        LangGraph StateGraph builder (Phase 1/2/3)
    memory/       SentenceTransformers embedding engine
    state/        ARIAState TypedDict — shared across all agents
    store/        Experience store
    tools/        calculator · web_search · write_file · read_file
    utils/        Rich CLI display helpers
  api/
    main.py       FastAPI application (Runtime API)
    schemas.py    Pydantic request/response models
  data/
    synthetic/    ARIA-Bench — 1,200 labeled synthetic runs (6 classes × 200)
    compiled/     Compiled DSPy Diagnostician + XGBoost model
    realbench/    50 real agent task results with human labels
  research/
    taxonomy_v1.md              Formal failure taxonomy with signal mapping
    confusion_audit_v1.md       Pre-training boundary analysis
    misclassification_analysis_v1.md  Post-training error analysis
    research_findings_v1.md     Documented research findings
    gap_failure_analysis_v1.md  Gap label deep audit (requirement failure sub-types)
    realbench_analysis_v1.md    Real-world distribution analysis
  scripts/
    generate_synthetic_data.py  ARIA-Bench generator (Task Ontology)
    run_realbench.py            Real task batch runner
    review_realbench.py         Human labeling interface
    analyze_realbench.py        Distribution + gap analysis
    audit_gaps.py               Gap label deep audit
    compare_validation_runs.py  Before/after comparison
    validate_critic_v2.py       Critic v2 validation on gap runs
    data_stats.py               Synthetic data quality report
  main.py         CLI entry point
```

---

## Roadmap

| Phase | Status | Description |
|---|---|---|
| Research Cycle 0 | ✅ Complete | Taxonomy + ARIA-Bench + DSPy Diagnostician |
| Research Cycle 0.5 | ✅ Complete | Confusion audit — class boundary validation |
| Research Cycle 0.75 | ✅ Complete | Misclassification analysis — goal_misalignment ontology finding |
| Research Cycle 1 | ✅ Complete | Real-world trace validation (50 tasks, human labels) |
| Research Cycle 1.25 | ✅ Complete | Gap failure analysis — requirement omission sub-types |
| Critic v2 | ✅ Complete | Requirement-aware evaluation (8% → 42% agreement) |
| **ARIA Alpha** | 🔄 In Progress | Runtime API + trace ingestion + failure reports |
| Research Cycle 2 | Planned | Multi-label diagnosis (mechanism + outcome simultaneously) |
| ARIA Dashboard | Planned | Failure distribution visualization |
| Taxonomy v2 | Planned | Two-layer: mechanism layer + outcome layer |
| XGBoost on real data | Planned | Train classifier on human-labeled RealBench |
| Paper | Targeting | ICLR 2026 workshop / ACL 2026 System Demonstrations |

---

## Research Publications

Targeting: ICLR 2026 Workshop on Reliable and Responsible Foundation Models
or ACL 2026 System Demonstrations track

Core claims:
1. A 5-class behavioral failure taxonomy for tool-using LLM agents, empirically validated
2. Goal misalignment functions as an outcome-level label, not a mechanism-level label
3. Holistic evaluation overestimates agent success — requirement-aware evaluation is necessary
