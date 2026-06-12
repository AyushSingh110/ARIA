# REST API Reference

Start the server:

```bash
cd backend
uvicorn api.main:app --port 8000
```

Interactive docs (Swagger UI): **http://localhost:8000/docs**

> ⚠️ The API has no built-in authentication — run it on localhost or behind a reverse proxy with auth. See [SECURITY.md](../SECURITY.md).

---

## `POST /diagnose`

Diagnose a pre-computed agent trace.

### Request

```json
{
  "task_description": "Find the population of France and save it to population.txt",
  "tool_calls": [
    {
      "tool_name": "web_search",
      "tool_args": {"query": "population of France"},
      "tool_result": "67.8 million (2024 estimate)",
      "turn": 0
    }
  ],
  "final_output": "The population of France is 67.8 million."
}
```

| Field | Type | Required | Notes |
|---|---|---|---|
| `task_description` | string | ✅ | |
| `tool_calls` | array | — | `turn` optional (-1 = auto by position) |
| `final_output` | string | — | |

### Response `200`

```json
{
  "task_id": "d-20260612-103045-a1b2",
  "failure_class": "goal_misalignment",
  "confidence": 0.84,
  "reasoning": "...",
  "manifestation": "requirement omission",
  "suggested_action": "...",
  "requirement_satisfaction": 0.5,
  "requirements": ["find the population of France", "save it to population.txt"],
  "requirements_satisfied": [true, false],
  "evidence": ["Requirement not satisfied: 'save it to population.txt'"],
  "observer_flags": [],
  "critic_scores": {"correctness": 4.0, "completeness": 2.0, "efficiency": 5.0, "safety": 5.0, "overall": 3.5, "pass_fail": false},
  "executor_turn_count": 1,
  "trace_summary": "turn 0: web_search({\"query\": \"population of France\"}) -> 67.8 million..."
}
```

`failure_class` is `null` for clean runs. Every diagnosis is persisted as a potential training record.

---

## `POST /diagnose/batch`

Diagnose multiple traces in one call.

### Request

```json
{ "traces": [ {<DiagnoseRequest>}, {<DiagnoseRequest>}, ... ] }
```

### Response `200`

```json
{
  "results": [ {<DiagnosisResponse>}, ... ],
  "total": 12,
  "failure_distribution": {"goal_misalignment": 5, "none": 6, "tool_misuse": 1}
}
```

---

## `POST /run`

Run a task through the **full pipeline** (the agent actually executes it with tools), then diagnose the run.

### Request

```json
{
  "task": "Calculate compound interest on $10,000 at 5% for 3 years",
  "task_class": "reasoning",
  "max_turns": 5
}
```

### Response `200`

Same shape as `/diagnose`, plus the executor's actual output in `trace_summary`.

---

## `POST /feedback`

Submit a human correction for a previous diagnosis. Corrections become labeled training data for the next Diagnostician recompile — this is how ARIA improves through usage.

### Request

```json
{
  "task_id": "d-20260612-103045-a1b2",
  "aria_correct": false,
  "human_label": "tool_misuse",
  "notes": "The search tool errored twice; ARIA missed it."
}
```

| Field | Type | Notes |
|---|---|---|
| `task_id` | string | From the diagnosis response |
| `aria_correct` | bool | Was ARIA's label right? |
| `human_label` | string | Required when `aria_correct=false` |
| `notes` | string | Optional reasoning |

### Response `200`

```json
{ "task_id": "d-...", "recorded": true, "message": "Feedback saved as training data." }
```

---

## `GET /dashboard`

Aggregate statistics powering the React dashboard.

### Response `200`

```json
{
  "total_runs": 147,
  "class_distribution": {"goal_misalignment": 51, "none": 60, "tool_misuse": 14},
  "class_distribution_pct": {"goal_misalignment": 34.7},
  "avg_confidence": 0.81,
  "avg_requirement_satisfaction": 0.62,
  "pass_rate": 0.41,
  "most_common_failure": "goal_misalignment",
  "labeled_runs": 50,
  "human_agreement_rate": 0.78,
  "recent_failures": [ {...last 10 failure records...} ]
}
```

---

## `GET /health`

```json
{ "status": "ok" }
```

---

## Status codes

| Code | Meaning |
|---|---|
| 200 | Success |
| 422 | Validation error (malformed request body) |
| 500 | Pipeline error (often an upstream LLM failure — check server logs) |

## Notes for production

- LLM calls make `/diagnose` take 3–15 s; size client timeouts accordingly (SDK default: 120 s).
- Groq free tier is 12,000 tokens/minute — for batch workloads add delays or upgrade the tier.
- Records are stored as JSON files under `backend/data/api_runs/` by default.
