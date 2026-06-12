# Contributing to ARIA

Thanks for your interest in improving ARIA! This document covers everything you need to get a change merged.

---

## Ways to contribute

| Contribution | Where to start |
|---|---|
| Report a bug | [Open an issue](https://github.com/AyushSingh110/ARIA/issues) with a minimal repro |
| Fix a bug / small improvement | Open a PR directly |
| New feature or behavior change | Open an issue first to discuss the design |
| New framework adapter (CrewAI, AutoGen, …) | See [Adding an adapter](#adding-a-framework-adapter) below — these are very welcome |
| Improve docs | PRs welcome, no issue needed |
| Research discussion (taxonomy, evaluation method) | Open an issue with the `research` label |

---

## Development setup

```bash
git clone https://github.com/AyushSingh110/ARIA.git
cd ARIA/backend

# Python 3.10+ required
pip install -e ".[api,dev]"
cp .env.example .env        # add your GROQ_API_KEY

# Optional: local models via Ollama (default executor/critic)
# https://ollama.com — then: ollama pull llama3.1:8b
```

Verify your setup:

```bash
python -c "from aria.sdk import diagnose; print('OK')"
pytest tests/ -x
```

Frontend (only needed for dashboard changes):

```bash
cd frontend
npm install
npm run dev          # http://localhost:5173, proxies /api to :8000
```

---

## Project layout

```
backend/
  aria/agents/        # the 7 diagnostic agents — most logic lives here
  aria/dspy_programs/ # DSPy Diagnostician (compiled program loaded at runtime)
  aria/graph/         # LangGraph pipeline wiring
  aria/sdk.py         # public SDK surface — keep stable
  adapters/           # framework adapters (LangGraph, OpenAI, ...)
  api/                # FastAPI runtime
  scripts/            # benchmarks, labeling, analysis (not shipped in the package)
frontend/             # React dashboard
docs/                 # user-facing documentation
```

---

## Pull request guidelines

1. **One change per PR.** Small PRs get reviewed fast.
2. **Run the linter and tests** before pushing:
   ```bash
   ruff check backend/
   pytest backend/tests/ -x
   ```
3. **Don't break the SDK surface.** `aria.sdk.diagnose()` / `diagnose_remote()` / `run_task()` signatures and the failure-report schema are public API — changes need discussion first.
4. **Match the existing style.** Type hints, docstrings on public functions, `rich` console for user-facing output.
5. **No data files in PRs.** Benchmark data, traces, and labeled datasets are private research assets and are gitignored — PRs containing them will be closed.

---

## Adding a framework adapter

Adapters convert a framework's native trace into ARIA's canonical format. Look at [`backend/adapters/openai_adapter.py`](backend/adapters/openai_adapter.py) as the template. An adapter needs to produce:

```python
tool_calls = [
    {"tool_name": str, "tool_args": dict, "tool_result": str},
    ...
]
```

and then call `aria.sdk.diagnose(task, tool_calls, final_output)`. Include:

- `diagnose_<framework>_trace(...)` function with a docstring example
- A test with a small synthetic trace in `backend/tests/`
- A short section in `docs/adapters.md`

---

## The failure taxonomy (context for contributors)

ARIA classifies agent failures into five classes. If your change touches classification logic, understand these first:

| Class | Layer | Signal |
|---|---|---|
| `prompt_drift` | mechanism | embedding drift across turns |
| `tool_misuse` | mechanism | tool errors / error loops (requires actual error evidence) |
| `context_overflow` | mechanism | repetition of completed steps |
| `hallucination_loop` | mechanism | claims without grounding; contradicted by independent search |
| `goal_misalignment` | outcome | requirements not satisfied despite "completion" |

Classification = DSPy program → deterministic disambiguation rules → Critic v3 grounding override. The deterministic rules in [`backend/aria/agents/diagnostician.py`](backend/aria/agents/diagnostician.py) encode hard-won findings from human-labeled data — don't relax them without benchmark evidence.

---

## Reporting bugs

Include:

1. What you ran (exact command or code)
2. What happened vs. what you expected
3. Python version, OS, and `pip show ariadx` version
4. If a diagnosis is wrong: the trace JSON (redact anything sensitive) and what label you expected

---

## Code of Conduct

Be respectful. See [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).

## License

By contributing, you agree your contributions are licensed under [Apache 2.0](LICENSE).
