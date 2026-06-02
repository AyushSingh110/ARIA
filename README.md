# ARIA — Autonomous Reflective Intelligence Architecture

A closed-loop multi-agent system where agents autonomously detect behavioral failures, classify the root cause, rewrite the broken component, and store the correction as reusable experience — without human intervention.

**Current phase:** Phase 1 — Orchestrator + Executor + Observer

---

## Quick start

```bash
# 1. Clone and create a virtualenv
python -m venv .venv && .venv\Scripts\activate   # Windows
pip install -r requirements.txt

# 2. Configure environment
cp .env.example .env
# Edit .env — add GROQ_API_KEY (or set ORCHESTRATOR_PROVIDER=ollama)

# 3. If using Ollama, pull the model
ollama pull llama3.1:8b

# 4. Run a task
python main.py "Calculate compound interest on $10,000 at 5% annual rate for 3 years"

# 5. Validate Phase 1 Observer
python scripts/inject_failure.py

# 6. Run tests
pytest tests/
```

---

## Project structure

```
aria/
  config/       Pydantic Settings (env-driven)
  state/        ARIAState TypedDict — shared across all 7 agents
  agents/       orchestrator · executor · observer  (Phase 1)
  tools/        calculator · web_search · file_ops
  memory/       SentenceTransformers embedding engine
  graph/        LangGraph StateGraph builder
  utils/        Rich CLI display helpers
logs/           Structured JSON observer logs (git-ignored)
scripts/        inject_failure.py — Phase 1 validation
tests/          Tool unit tests (no API keys required)
```

See [DOCUMENTATION.md](DOCUMENTATION.md) for full architecture, implementation flow, and phase completion notes.

---

## Roadmap

| Phase | Agents | Status |
|---|---|---|
| 1 | Orchestrator + Executor + Observer | ✅ Complete |
| 2 | + Critic + Diagnostician | Pending |
| 3 | + Refiner + Validator | Pending |
| 4 | FAISS + MongoDB experience store | Pending |
| 5 | Benchmark 500 runs + paper | Pending |
