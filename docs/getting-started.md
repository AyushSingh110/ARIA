# Getting Started

This guide takes you from zero to your first agent diagnosis in about 5 minutes.

---

## What ARIA does

You give ARIA an agent trace (the task, the tool calls the agent made, and its final answer). ARIA tells you:

- **Did the agent fail?** — and with what confidence
- **How did it fail?** — one of five behavioral failure classes
- **Which requirements were missed?** — a per-requirement checklist
- **What should you fix?** — a concrete suggested action

---

## Installation

### Option A — from PyPI (recommended)

```bash
pip install ariadx          # core SDK
pip install "ariadx[api]"   # + REST API server
```

### Option B — from source

```bash
git clone https://github.com/AyushSingh110/ARIA.git
cd ARIA/backend
pip install -e ".[api]"
```

### Configuration

ARIA needs at least one LLM provider. Create a `.env` file (or copy `backend/.env.example`):

```ini
# Required — free key at https://console.groq.com
GROQ_API_KEY=gsk_...
GROQ_MODEL=llama-3.3-70b-versatile

# Provider routing (groq = cloud, ollama = local)
ORCHESTRATOR_PROVIDER=groq
EXECUTOR_PROVIDER=ollama        # set to groq if you don't run Ollama
CRITIC_PROVIDER=ollama          # set to groq if you don't run Ollama

# Critic v3 — independent factual grounding (hallucination detection)
GROUNDING_ENABLED=true
```

> **No Ollama?** Set `EXECUTOR_PROVIDER=groq` and `CRITIC_PROVIDER=groq` — everything runs on the Groq free tier. Ollama (with `llama3.1:8b`) just keeps executor/critic calls local and free of rate limits.

---

## Your first diagnosis

```python
from aria.sdk import diagnose

report = diagnose(
    task="Find the population of France and save it to population.txt",
    tool_calls=[
        {"tool_name": "web_search",
         "tool_args": {"query": "population of France"},
         "tool_result": "67.8 million (2024 estimate) — INSEE"},
    ],
    final_output="The population of France is 67.8 million.",
)

print(report["failure_class"])              # "goal_misalignment"
print(report["requirement_satisfaction"])   # 0.5
print(report["requirements"])               # ["find the population of France",
                                            #  "save it to population.txt"]
print(report["requirements_satisfied"])     # [True, False]  ← never saved the file
print(report["suggested_action"])
```

The agent *found* the population but never *saved the file* — a classic `goal_misalignment`: the task looks complete, but a requirement was silently dropped. This is the most common failure mode in real-world traces (36% in our benchmark).

---

## Diagnosing a clean run

```python
report = diagnose(
    task="What is 2 to the power of 10?",
    tool_calls=[
        {"tool_name": "calculator",
         "tool_args": {"expression": "2**10"},
         "tool_result": "1024"},
    ],
    final_output="2 to the power of 10 is 1024.",
)
print(report["failure_class"])   # None — no failure detected
```

`failure_class` is `None` (or `"none"` over the REST API) when the run is clean.

---

## Using the REST API instead

```bash
cd backend
uvicorn api.main:app --port 8000
```

```bash
curl -X POST http://localhost:8000/diagnose \
  -H "Content-Type: application/json" \
  -d '{
    "task_description": "Find the population of France and save it to population.txt",
    "tool_calls": [{"tool_name": "web_search",
                    "tool_args": {"query": "population of France"},
                    "tool_result": "67.8 million"}],
    "final_output": "The population of France is 67.8 million."
  }'
```

Interactive Swagger docs: http://localhost:8000/docs
Full endpoint reference: [api-reference.md](api-reference.md)

---

## The dashboard

```bash
cd frontend
npm install
npm run dev      # http://localhost:5173
```

The dashboard shows live failure distribution, recent failures, and a diagnose panel where you can paste a trace and get a diagnosis with one click. Every diagnosis you correct via the feedback buttons becomes training data — ARIA improves through usage.

---

## Coming from LangGraph or OpenAI?

Skip manual trace conversion — use the adapters:

```python
# LangGraph message history
from adapters.langgraph_adapter import diagnose_langgraph_trace
report = diagnose_langgraph_trace(messages, task_description="...")

# OpenAI Assistants run steps / chat completions with tool_calls
from adapters.openai_adapter import diagnose_openai_trace
report = diagnose_openai_trace(run_steps, task_description="...")
```

See [adapters.md](adapters.md) for details.

---

## Next steps

- [SDK reference](sdk-reference.md) — all three SDK functions in detail
- [API reference](api-reference.md) — REST endpoints, request/response schemas
- [Failure taxonomy](failure-taxonomy.md) — what the five classes mean and how they're detected
- [Architecture](architecture.md) — how the diagnostic pipeline works inside
