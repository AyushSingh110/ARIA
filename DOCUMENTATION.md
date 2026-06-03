# ARIA — Technical Documentation

> This file is updated after each phase completion. It is the authoritative reference for architecture, implementation decisions, and inter-agent contracts.

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Phase 1 — Implementation](#2-phase-1--implementation)
   - [Architecture Diagram](#21-architecture-diagram)
   - [Shared State Schema](#22-shared-state-schema-ariastate)
   - [Agent Contracts](#23-agent-contracts)
   - [Tool Specifications](#24-tool-specifications)
   - [Graph Topology](#25-graph-topology)
   - [Observer Anomaly Detectors](#26-observer-anomaly-detectors)
   - [Embedding Engine](#27-embedding-engine)
   - [Configuration Reference](#28-configuration-reference)
   - [Observer Log Format](#29-observer-log-format)
   - [Running and Validating](#210-running-and-validating)
3. [Phase 2 - Implementation (Critic + Diagnostician)](#phase-2-implementation)
4. [Phase 3 - Implementation (Refiner + Validator)](#phase-3-implementation)
5. [Phase 4 — Plan (Experience Store)](#5-phase-4--plan)
6. [Phase 5 — Plan (Benchmark + Paper)](#6-phase-5--plan)
7. [Failure Taxonomy Reference](#7-failure-taxonomy-reference)
8. [Tech Stack Decisions](#8-tech-stack-decisions)

---

## 1. System Overview

ARIA is a 7-agent closed-loop system. Agents are nodes in a LangGraph `StateGraph` with cyclic edges. All agents share a single typed state object (`ARIAState`) that flows through the graph. No agent communicates with another directly — all coordination is through state mutations.

**The 7 agents and their phases:**

| Agent | Phase | Role |
|---|---|---|
| Orchestrator | 1 | Task decomposition, routing |
| Executor | 1 | Tool-calling task execution |
| Observer | 1 | Structured logging, anomaly detection |
| Critic | 2 | Independent output quality scoring |
| Diagnostician | 2 | Failure root cause classification |
| Refiner | 3 | Rewrite broken agent components |
| Validator | 3 | Test refined components, commit to experience store |

**Core research contribution:** A 5-class failure taxonomy (prompt drift, tool misuse, context overflow, goal misalignment, hallucination loop) empirically validated on 500+ synthetic agent runs.

---

## 2. Phase 1 — Implementation

**Status: Complete**
**Completion date: 2026-06-02**

Phase 1 delivers the "structured failure capture" system. It proves two architectural assumptions before adding complexity:
1. LangGraph cyclic graphs route state correctly between nodes with full fidelity.
2. Observer logs are rich enough for a future classifier (Phase 2 Diagnostician) to diagnose failures from them alone.

### 2.1 Architecture Diagram

```
User Input (task description)
         │
         ▼
  ┌─────────────────┐
  │   ORCHESTRATOR  │  ← LLM (Groq / Ollama)
  │  - Decomposes   │    Classifies task
  │  - Sets subtask │    Generates subtask list
  └────────┬────────┘    Embeds goal description
           │
           ▼ (always)
  ┌─────────────────┐
  │    EXECUTOR     │  ← LLM (Ollama / Groq)
  │  - Tool calling │    Runs tool loop up to max_turns
  │  - Builds trace │    Records every tool call + result
  └────────┬────────┘    Embeds each turn for drift detection
           │
           ▼ (always)
  ┌─────────────────┐
  │    OBSERVER     │  ← No LLM
  │  - 4 detectors  │    Cosine drift, repetition,
  │  - JSON log     │    budget warning, error loops
  │  - Flag scorer  │    Writes logs/{task_id}.json
  └────────┬────────┘
           │
     ┌─────┴──────┐
     │ Conditional│
     └─────┬──────┘
  more     │      task
 subtasks  │    complete
     │     │      │
     ▼     │      ▼
 Orchestrator   END
 (cycle back)
```

### 2.2 Shared State Schema (`ARIAState`)

Defined in `aria/state/schema.py`. Key design decisions:

- **`executor_trace` and `observer_flags` use `Annotated[list, operator.add]`** — LangGraph reducer that appends new items rather than overwriting. This allows accumulation across multiple subtask cycles.
- **`task_description_embedding`** — embedded once by Orchestrator, reused by Observer for all drift calculations. No re-embedding on loops.
- **`goal_embedding_history`** — populated by Executor, one entry per turn. Observer computes drift against the goal embedding.
- **Phase 2/3 fields are present in schema but always `None` in Phase 1.** This ensures the schema is stable across all phases and no migration is needed.

**Full schema:**

```python
class ARIAState(TypedDict):
    # Task identity
    task_id: str                           # UUID per run
    task_description: str                  # raw user input
    task_class: str                        # Orchestrator-classified
    task_description_embedding: list[float] # L2-normalised embedding

    # Decomposition
    subtasks: list[dict]                   # [{id, description, expected_tools, success_criteria}]
    current_subtask_index: int
    active_subtask: Optional[dict]

    # Executor (append-only via operator.add reducer)
    executor_output: Optional[str]
    executor_trace: Annotated[list[ExecutorTraceEntry], operator.add]
    executor_turn_count: int
    goal_embedding_history: list[list[float]]  # one per turn
    drift_scores: list[float]                  # cosine distance per turn

    # Observer (append-only)
    observer_flags: Annotated[list[ObserverFlag], operator.add]
    anomaly_detected: bool
    anomaly_severity: float                # max flag signal_value
    observer_log_path: Optional[str]

    # Phase 2 (null in Phase 1)
    critic_scores: Optional[CriticScores]
    failure_class: Optional[str]
    failure_manifestation: Optional[str]
    diagnosis_confidence: Optional[float]
    diagnosis_reasoning: Optional[str]

    # Phase 3 (null in Phase 1)
    refinement: Optional[RefinementRecord]
    refinement_applied: bool
    post_refinement_scores: Optional[CriticScores]
    delta_score: Optional[float]
    committed_to_store: bool
    experience_record_id: Optional[str]

    # Control flow
    retry_count: int
    max_retries: int
    escalate: bool
    current_phase: Literal["decompose", "execute", "observe", ...]

    # LangGraph message channel
    messages: Annotated[list, add_messages]

    # Metadata
    run_start_time: str                    # ISO-8601
    total_tokens_used: int
    api_calls_groq: int
    api_calls_ollama: int
```

### 2.3 Agent Contracts

Each agent is a **pure function** `(ARIAState) → dict`. It returns only the fields it mutates. LangGraph merges the returned dict into the current state.

#### Orchestrator

**Input fields read:** `task_description`, `api_calls_groq`, `api_calls_ollama`

**Output fields written:**
```python
{
    "task_class": str,
    "subtasks": list[dict],
    "current_subtask_index": 0,
    "active_subtask": dict,
    "task_description_embedding": list[float],
    "current_phase": "execute",
    "api_calls_groq": int,
    "api_calls_ollama": int,
}
```

**LLM prompt strategy:** Single system prompt + single human message. Structured JSON output enforced. Regex strips markdown fences before JSON parse. Fallback to passthrough subtask on parse error.

**Model:** Configurable via `ORCHESTRATOR_PROVIDER` (default: `groq`). Uses `temperature=0` for deterministic decomposition.

#### Executor

**Input fields read:** `active_subtask`, `api_calls_groq`, `api_calls_ollama`, `total_tokens_used`

**Output fields written:**
```python
{
    "executor_output": str,
    "executor_trace": list[ExecutorTraceEntry],   # appended
    "executor_turn_count": int,
    "goal_embedding_history": list[list[float]],
    "current_phase": "observe",
    "total_tokens_used": int,
    "api_calls_groq": int,
    "api_calls_ollama": int,
}
```

**Tool calling loop:**
1. Build messages: `[SystemMessage, HumanMessage(subtask)]`
2. Call `llm.bind_tools(EXECUTOR_TOOLS).invoke(messages)`
3. If no `tool_calls`: extract final answer, append trace entry, break
4. If `tool_calls`: dispatch each call via `_TOOL_MAP[tool_name].invoke(args)`, append `ToolMessage`, embed the primary tool call for drift tracking, loop
5. Hard limit: `EXECUTOR_MAX_TURNS` (default 10)

**Trace entry per turn:** `{turn, tool_name, tool_args, tool_result, llm_output, latency_ms, token_count}`

**Model:** Configurable via `EXECUTOR_PROVIDER` (default: `ollama`). Llama 3.1 8B locally.

#### Observer

**Input fields read:** `executor_trace`, `task_description_embedding`, `goal_embedding_history`, `executor_turn_count`, `subtasks`, `current_subtask_index`

**Output fields written:**
```python
{
    "observer_flags": list[ObserverFlag],         # appended
    "anomaly_detected": bool,
    "anomaly_severity": float,
    "drift_scores": list[float],
    "observer_log_path": str,
    "current_phase": "complete" | "decompose",
    "current_subtask_index": int,
    "active_subtask": dict,
}
```

**No LLM call.** Pure signal computation + JSON log writing.

### 2.4 Tool Specifications

All tools are `@tool`-decorated LangChain functions. They return `str` always (never raise to the LLM).

| Tool | Module | Description | Phase 1 impl |
|---|---|---|---|
| `calculator` | `aria/tools/calculator.py` | Safe AST-based math evaluator | Real — no external calls |
| `write_file` | `aria/tools/file_ops.py` | Write to sandboxed `workspace/` dir | Real — filesystem |
| `read_file` | `aria/tools/file_ops.py` | Read from sandboxed `workspace/` dir | Real — filesystem |
| `web_search` | `aria/tools/web_search.py` | Search the web | Mock — keyword-matched canned responses |

**Calculator safety:** Uses Python `ast` module to parse expressions. Only allows `ast.BinOp`, `ast.UnaryOp`, `ast.Constant`, `ast.Name` (whitelisted only), and `ast.Call` (whitelisted functions only). No `eval()` — zero code execution risk.

**File ops sandboxing:** All paths are resolved against a `workspace/` directory. `Path.resolve()` is compared against workspace root to prevent `../../etc/passwd` style traversal. Symlinks are implicitly blocked by resolve semantics.

**Web search upgrade path:** `_fetch_results()` in `web_search.py` is the only function to replace when wiring a real search API. Signature and return type are unchanged.

### 2.5 Graph Topology

```python
graph = StateGraph(ARIAState)
graph.add_node("orchestrator", orchestrator_node)
graph.add_node("executor", executor_node)
graph.add_node("observer", observer_node)

graph.add_edge(START, "orchestrator")
graph.add_edge("orchestrator", "executor")
graph.add_edge("executor", "observer")
graph.add_conditional_edges(
    "observer",
    _route_after_observer,      # returns "orchestrator" | "__end__"
    {"orchestrator": "orchestrator", "__end__": END}
)
```

**Routing logic (`_route_after_observer`):**
- `state["escalate"] == True` → `END`
- `state["current_phase"] == "complete"` → `END`
- Otherwise (more subtasks) → `"orchestrator"` (cycle)

**Why not DAG:** Phase 2 adds Critic → Diagnostician → Refiner → Validator, all cycling back through Executor on failed refinements. The cyclic structure is essential from the start. LangGraph's `StateGraph` natively supports cycles; LangChain's original `Chain` does not.

### 2.6 Observer Anomaly Detectors

Four detectors run sequentially in `observer_node`. Each returns a list of `ObserverFlag` objects.

#### Detector 1: Prompt Drift
```
Signal: cosine_distance(goal_embedding, turn_embedding) > ANOMALY_DRIFT_THRESHOLD
Threshold: 0.45 (configurable via env)
Severity: the raw cosine distance value (0.0–2.0, but normalised embeddings cap at 1.0)
Flag type: "prompt_drift"
```
Computed for every turn in `goal_embedding_history`. Multiple turns can each generate a flag. The Diagnostician (Phase 2) will use the *sequence* of drift scores (monotonically increasing = drift, sudden spike = context overflow) to distinguish classes.

#### Detector 2: Tool Repetition
```
Signal: same (tool_name, MD5(tool_args)) appears more than once in trace
Severity: 0.8 (fixed — repetition is a strong signal)
Flag type: "tool_repetition"
```
Does not flag `__llm__` entries (pure LLM outputs without tool calls).

#### Detector 3: Turn Budget Warning
```
Signal: executor_turn_count / EXECUTOR_MAX_TURNS >= 0.8
Severity: the actual ratio (0.8–1.0)
Flag type: "turn_budget_warning"
```
An agent consuming 80%+ of its turn budget is likely struggling, not efficiently completing.

#### Detector 4: Tool Error Loop
```
Signal: trace[i].tool_result contains "Error" AND trace[i].tool_name == trace[i+1].tool_name
Severity: 0.9 (fixed — error retry without change = tool misuse pattern)
Flag type: "tool_error_loop"
```
Catches the pattern: tool fails → agent retries with same tool → same failure.

### 2.7 Embedding Engine

`aria/memory/embeddings.py` — singleton via `@lru_cache(maxsize=1)`.

**Model:** `BAAI/bge-small-en-v1.5` — 33M parameters, 384-dim embeddings, entirely local.
- All embeddings are L2-normalised at encode time (`normalize_embeddings=True`)
- Cosine similarity = dot product (no additional normalization needed at query time)
- `cosine_distance(a, b) = 1.0 - dot(a, b)` — range [0, 2], practical range [0, 1]

**First run:** Model downloads from HuggingFace (~120MB). Subsequent runs use local cache. Set `SENTENCE_TRANSFORMERS_HOME` env var to control cache location.

### 2.8 Configuration Reference

All settings are in `.env` (copy from `.env.example`). Loaded via Pydantic Settings — type-validated, with defaults.

| Variable | Default | Description |
|---|---|---|
| `GROQ_API_KEY` | _(required if using groq)_ | Groq API key |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | Groq model ID |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_MODEL` | `llama3.1:8b` | Ollama model tag |
| `ORCHESTRATOR_PROVIDER` | `groq` | `groq` or `ollama` |
| `EXECUTOR_PROVIDER` | `ollama` | `groq` or `ollama` |
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | SentenceTransformers model |
| `LOG_DIR` | `logs` | Directory for observer JSON logs |
| `MAX_RETRIES` | `3` | Max refinement retries (Phase 3) |
| `EXECUTOR_MAX_TURNS` | `10` | Max tool-calling turns per subtask |
| `ANOMALY_DRIFT_THRESHOLD` | `0.45` | Cosine distance above which drift is flagged |

### 2.9 Observer Log Format

Every run writes `logs/{task_id}.json`. This is the primary artifact Phase 2 Diagnostician will consume.

```json
{
  "schema_version": "1.0",
  "task_id": "uuid",
  "task_description": "...",
  "task_class": "code_generation",
  "run_start_time": "2026-06-02T10:00:00Z",
  "run_end_time": "2026-06-02T10:00:45Z",
  "executor_trace": [
    {
      "turn": 0,
      "tool_name": "calculator",
      "tool_args": {"expression": "2**10"},
      "tool_result": "1024",
      "llm_output": "I will use the calculator…",
      "latency_ms": 312,
      "token_count": 45
    }
  ],
  "executor_turn_count": 2,
  "executor_output": "FINAL ANSWER: 2^10 = 1024",
  "observer_flags": [
    {
      "flag_type": "prompt_drift",
      "signal_value": 0.512,
      "turn": 3,
      "description": "Turn 3: cosine distance from goal = 0.5120…"
    }
  ],
  "drift_scores": [0.02, 0.05, 0.18, 0.51],
  "goal_embedding_available": true,
  "turn_embeddings_captured": 4,
  "anomaly_detected": true,
  "anomaly_severity": 0.512,
  "summary": {
    "total_turns": 4,
    "tools_called": ["calculator", "write_file"],
    "unique_tools": ["calculator", "write_file"],
    "flag_types": ["prompt_drift"]
  }
}
```

### 2.10 Running and Validating

**Install:**
```bash
pip install -r requirements.txt
cp .env.example .env   # then edit GROQ_API_KEY
```

**Run a task:**
```bash
python main.py "Calculate compound interest on $10,000 at 5% for 3 years"
python main.py "Search for Python facts and write them to facts.txt" --task-class web_research
python main.py "What is sqrt(2) squared?" --verbose
```

**Validate Phase 1 (no API keys needed):**
```bash
python scripts/inject_failure.py
python scripts/inject_failure.py --scenario prompt_drift
```

**Run unit tests:**
```bash
pytest tests/ -v
```

**Phase 1 success criterion:** Running `inject_failure.py --scenario all` passes all three scenarios. This means Observer's log structure captures the correct signal for each failure class, proving Phase 2 Diagnostician has sufficient data to classify from.

---

## Phase 2 Implementation

**Status: Complete**
**Completion date: 2026-06-02**

### Architecture change

Phase 2 inserts two new nodes between Observer and END. The graph is now fully linear per subtask — routing is handled by Diagnostician, not Observer.

```text
orchestrator → executor → observer → critic → diagnostician → (loop | END)
```

Observer no longer decides routing. It outputs `current_phase = "critique"` and passes straight to Critic. Diagnostician reads subtask index and decides whether to loop back or end.

### New files

| File | Purpose |
|---|---|
| `aria/agents/critic.py` | Scoring agent — 4-dimension quality assessment |
| `aria/agents/diagnostician.py` | Classification agent — 5-class failure taxonomy |
| `aria/dspy_programs/diagnostician.py` | DSPy Signature + Module definitions |
| `aria/classifiers/failure_classifier.py` | XGBoost wrapper (trains in Phase 5) |
| `scripts/generate_synthetic_data.py` | Programmatic training data generator |
| `scripts/compile_diagnostician.py` | Offline DSPy BootstrapFewShot compilation |

### Critic agent contract

**Input fields read:** `task_description`, `executor_output`, `executor_trace`, `api_calls_*`

**Output fields written:**

```python
{
    "critic_scores": CriticScores,   # correctness/completeness/efficiency/safety/overall/pass_fail
    "current_phase": "diagnose",
    "api_calls_groq": int,
    "api_calls_ollama": int,
}
```

**Scoring weights:** correctness×0.4 + completeness×0.3 + efficiency×0.2 + safety×0.1

**Pass/fail threshold:** overall ≥ 3.5

**Model:** `CRITIC_PROVIDER` env var (default: `ollama`). Completely decoupled from Observer — receives only task description + executor output, not observer flags. This prevents circular influence on scoring.

### Diagnostician agent contract

**Input fields read:** `task_description`, `observer_flags`, `critic_scores`, `executor_trace`

**Output fields written:**

```python
{
    "failure_class": str | None,          # primary root cause or None
    "failure_manifestation": str | None,  # secondary pattern or None
    "diagnosis_confidence": float,
    "diagnosis_reasoning": str,
    "current_phase": "complete" | "decompose",
    "current_subtask_index": int,
    "active_subtask": dict,
    "api_calls_groq": int,
}
```

**Two-stage classification:**

1. XGBoost on numeric features (fast, local) — if trained and predicts a failure, uses as prior
2. DSPy ChainOfThought via Groq — primary classifier using full context
3. If XGBoost predicts failure but DSPy says `none`, the XGBoost signal overrides with confidence floor 0.6

**DSPy program lifecycle:**

- `DiagnoseFailure` Signature defines 4 inputs + 4 outputs
- `DiagnosticProgram` wraps it with `dspy.ChainOfThought`
- At startup, `diagnostician.py` tries to load `data/compiled/diagnostician.json`
- If not found: zero-shot (works fine, just less consistent)
- Compiled by running `scripts/compile_diagnostician.py` (offline, ~20 Groq API calls)

**DSPy is compile-time only.** `BootstrapFewShot` runs offline and writes optimized few-shot demonstrations into the saved JSON. At inference time, DSPy replays those demos — no optimization loop, no training cost per run.

### XGBoost classifier features

12 numeric features extracted from state by `FailureFeatureExtractor`:

```text
max_drift_score, mean_drift_score,
n_prompt_drift_flags, n_tool_repetition_flags,
n_tool_error_loop_flags, n_turn_budget_flags,
turn_ratio (used/max),
critic_correctness, critic_completeness, critic_efficiency,
critic_safety, critic_overall
```

Not trained until Phase 5 (needs 500 real run records). Until then `predict()` returns `None` and Diagnostician uses DSPy only.

### Synthetic data workflow

```bash
# Step 1: generate training data (no API calls, pure Python)
python scripts/generate_synthetic_data.py --per-class 100
# → data/synthetic/{prompt_drift,tool_misuse,...}.jsonl

# Step 2: compile DSPy program (calls Groq ~20 times, costs ~$0.01)
python scripts/compile_diagnostician.py --max-demos 4
# → data/compiled/diagnostician.json
```

Each synthetic example contains: `task_description`, `observer_flags` (JSON), `critic_scores` (JSON), `trace_summary`, `failure_class` (gold label), `failure_manifestation` (gold label).

The generator is deterministic with `--seed 42` for reproducibility.

### Running Phase 2

```bash
cd backend
python main.py "Calculate compound interest on $5000 at 7% for 5 years"
```

Full pipeline output now includes:

- Orchestrator: task class + subtask
- Executor: tool calls + trace
- Observer: anomaly flags + JSON log
- Critic: 4 dimension scores + pass/fail
- Diagnostician: failure class + confidence + reasoning

### Phase 2 success criterion

Run 5 tasks covering different failure classes injected via `inject_failure.py`. Diagnostician should classify each correctly with confidence > 0.6. If compiled DSPy program is in place, expect > 70% accuracy on the 20-example validation set printed by `compile_diagnostician.py`.

---

## Phase 3 Implementation

**Status: Complete**
**Completion date: 2026-06-02**

### Phase 3 architecture change

Phase 3 closes the loop. Failures now trigger autonomous rewriting and validation.

```text
orchestrator → executor → observer → critic → diagnostician
                                                    │
                              clean run ────────────┤
                                                    │ failure detected
                                              ┌─────▼──────┐
                                              │   REFINER  │ ← RAG from experience store
                                              └─────┬──────┘
                                                    │
                                              ┌─────▼──────┐
                                              │  VALIDATOR │ re-runs executor + critic
                                              └─────┬──────┘
                                                    │
                                  delta ≥ 0.3 ──────┤──── delta < 0.3 (retry → Refiner)
                                                    │
                                              commit to store → END
```

### Phase 3 new files

| File | Purpose |
|---|---|
| `aria/agents/refiner.py` | RAG-based component rewriter using Groq |
| `aria/agents/validator.py` | Re-runs executor + critic, computes delta, commits |
| `aria/store/experience_store.py` | Local JSON store (swapped for MongoDB in Phase 4) |

### Refiner agent contract

**Input fields read:** `failure_class`, `task_description`, `diagnosis_reasoning`, `task_class`, `retry_count`

**Output fields written:**

```python
{
    "refinement": RefinementRecord,   # target, original, refined, diff, semantic_distance
    "refinement_applied": False,      # Validator sets this to True before re-running executor
    "current_phase": "validate",
    "api_calls_groq": int,
}
```

**Failure → target mapping:**

| Failure class | Rewrite target |
|---|---|
| prompt_drift | system_prompt |
| tool_misuse | tool_schema (executor prompt tool section) |
| context_overflow | system_prompt |
| goal_misalignment | system_prompt |
| hallucination_loop | system_prompt |

**RAG retrieval:** Queries `LocalExperienceStore.retrieve_similar(failure_class, task_class, k=3)`. Returns the top-3 committed refinements sorted by `delta_score` descending. These are injected as few-shot examples into the Groq prompt.

**Constitutional guard:** After generation, cosine distance between original and refined is computed. If distance > 0.65: refinement is clamped to the original with a minimal appended correction. This prevents the Refiner from drifting too far and breaking the component.

### Validator agent contract

**Input fields read:** `refinement`, `critic_scores`, `active_subtask`, `retry_count`, `max_retries`

**Output fields written (committed path):**

```python
{
    "post_refinement_scores": CriticScores,
    "delta_score": float,
    "committed_to_store": True,
    "experience_record_id": str,
    "current_phase": "complete",
}
```

**Output fields written (retry path):**

```python
{
    "post_refinement_scores": CriticScores,
    "delta_score": float,
    "committed_to_store": False,
    "retry_count": int,
    "current_phase": "refine",
}
```

**Re-run mechanism:**

1. `_apply_refinement(state)` patches the state: resets `executor_trace`, `executor_output`, sets `refinement_applied = True`
2. Calls `executor_node(patched_state)` directly (not through graph) — executor reads the refined prompt
3. Calls `critic_node(merged_state)` to score the new output
4. Computes `delta = refined_overall - original_overall`

**Commit threshold:** delta ≥ 0.3

**Both committed and failed attempts are saved** to the experience store. Failed attempts (committed=False) still help the Refiner avoid repeating unsuccessful strategies.

### Experience store (Phase 3 — local JSON)

`aria/store/experience_store.py` — `LocalExperienceStore` backed by `data/experience_store.json`.

The public interface matches the MongoDB implementation planned for Phase 4:

```python
store.save(record: dict) → str              # record_id
store.retrieve_similar(failure_class, task_class, k) → list[dict]
store.all_committed() → list[dict]
store.count() → int
```

Each record contains:

```json
{
  "record_id": "uuid",
  "created_at": "iso-datetime",
  "task_class": "code_generation",
  "failure_class": "prompt_drift",
  "refinement_target": "system_prompt",
  "original_component": "...",
  "refined_component": "...",
  "diff": "...",
  "delta_score": 0.45,
  "committed": true,
  "original_critic_scores": {...},
  "refined_critic_scores": {...}
}
```

### Phase 3 routing (graph builder)

```python
# After diagnostician:
if failure_class and not pass_fail → "refiner"
if clean run and more subtasks    → "orchestrator"
if clean run and last subtask     → END

# After validator:
if committed or escalated         → END
else                              → "refiner"  # retry
```

### Running Phase 3

```bash
cd backend
python main.py "Your task here"
```

When a failure is detected, you will see all 7 agent panels in sequence. The experience store grows at `data/experience_store.json` with each committed refinement.

### Phase 3 success criterion

Run a task that triggers a failure (use `scripts/inject_failure.py` patterns as inspiration). Verify:

1. Diagnostician classifies the failure correctly
2. Refiner generates a rewritten component with semantic distance < 0.65
3. Validator re-runs executor with the refined prompt
4. If delta ≥ 0.3: record appears in `data/experience_store.json` with `committed: true`
5. On a second identical failure: Refiner retrieves the previous refinement as a few-shot example

---

## 5. Phase 4 — Plan

**Target: Weeks 10–11**

Add persistent experience store: ChromaDB (vector retrieval) + MongoDB Atlas (full records).

**Schema:** See research briefing document for full MongoDB schema.

**Key indexes:** `{task_class, failure_class, committed}`, `{created_at}`, `{delta_score}`.

**Experience transfer test:** Run 10 new tasks from each of 5 task classes. Verify that retrieval from the experience store improves delta scores vs. zero-shot refinement baseline.

---

## 6. Phase 5 — Plan

**Target: Weeks 12–14**

500-run benchmark across 5 task domains. Paper writing. PyPI SDK release.

**Paper target:** ICLR 2026 Workshop on Agentic AI, or ACL 2026 System Demonstrations.

**SDK:** `aria-sdk` on PyPI — exposes `ARIAGraph`, `ARIAState`, `FailureTaxonomy` as public API.

---

## 7. Failure Taxonomy Reference

| Class | Detection Signal | Refiner Target |
|---|---|---|
| Prompt drift | Monotonically increasing cosine distance from goal embedding | Rewrite executor system prompt with stronger goal anchoring |
| Tool misuse | Tool schema mismatch, error loops, wrong call sequence | Rewrite tool schema or add tool selection guidance to prompt |
| Context overflow | Self-contradiction score, tool repetition, constraint violation | Rewrite memory retrieval strategy; compress context |
| Goal misalignment | Critic score diverges from task spec alignment | Rewrite orchestrator decomposition or executor success criteria |
| Hallucination loop | Shadow model consensus disagreement, confident false repetition | Rewrite executor prompt with verification step requirement |

**Note on taxonomy structure:** Primary classes (root causes) are prompt drift, context overflow, goal misalignment. Secondary classes (behavioral manifestations) are tool misuse and hallucination loop. A single failure event can have one primary + one secondary class. The Diagnostician classifies both independently.

---

## 8. Tech Stack Decisions

| Component | Choice | Rationale |
|---|---|---|
| Orchestration | LangGraph 0.2 | Native cyclic graph support; not a DAG framework |
| LLM (cloud) | Groq / Llama-3.3-70B | Near-zero cost; ~$0.59/M tokens; fast inference |
| LLM (local) | Ollama / Llama 3.1 8B | Zero cost for executor/critic/validator |
| Prompt optimization | DSPy BootstrapFewShot | Offline compilation of Diagnostician prompt only — not runtime |
| Embeddings | SentenceTransformers BAAI/bge-small-en-v1.5 | Local, no API key, 384-dim, fast, good quality |
| Vector store | ChromaDB (Phase 4) | Stores vectors + metadata together; filters by field |
| Document store | MongoDB Atlas free tier | Flexible schema; 512MB free; compound indexes |
| Failure classifier | XGBoost + LLM ensemble (Phase 2) | XGBoost on numeric signals; LLM for reasoning |
| Sandbox | Docker + resource caps (Phase 2+) | E2B as cloud fallback |
| Backend | FastAPI on Cloud Run (Phase 5) | Serverless; free tier covers research load |
| Frontend | React/Vite (Phase 5) | Real-time dashboard; WebSocket for live agent state |
| Observability | LangSmith | Full agent trace visibility |

**DSPy clarification (important):** DSPy is a compile-time optimizer. `BootstrapFewShot` runs offline against a training set and produces an optimized prompt program. The compiled prompt is then used at inference time. DSPy is NOT used for runtime refinement — that is direct retrieval-augmented generation in the Refiner.
