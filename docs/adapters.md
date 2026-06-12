# Framework Adapters

Adapters convert another framework's native trace format into ARIA's canonical trace and send it to a running ARIA API server for diagnosis.

> Adapters use the REST API (`aria_url`), so start the server first: `uvicorn api.main:app --port 8000`. To diagnose in-process without a server, convert your trace to the `tool_calls` format manually and call `aria.sdk.diagnose`.

---

## LangGraph

```python
from adapters.langgraph_adapter import diagnose_langgraph_trace

# inside / after your LangGraph run:
result = diagnose_langgraph_trace(
    messages=state["messages"],                  # LangGraph message history
    task_description="Find the capital of France",
    aria_url="http://localhost:8000",
)
print(result["failure_class"], result["requirement_satisfaction"])
```

The adapter walks the message history, pairing `AIMessage.tool_calls` with the matching `ToolMessage` results, and uses the last AI message as `final_output`.

Lower-level conversion (no HTTP):

```python
from adapters.langgraph_adapter import langgraph_to_aria
req = langgraph_to_aria(messages, task_description)   # → DiagnoseRequest
```

---

## OpenAI

Supports two formats:

```python
from adapters.openai_adapter import diagnose_openai_trace

# Assistants API run steps
result = diagnose_openai_trace(
    data=run_steps,
    task_description="...",
    format="run_steps",
)

# Chat Completions messages with tool_calls
result = diagnose_openai_trace(
    data=chat_messages,
    task_description="...",
    format="chat",
)
```

Lower-level conversion: `openai_to_aria(data, task_description, format=...)`.

---

## Writing your own adapter

An adapter only needs to produce ARIA's canonical trace:

```python
tool_calls = [
    {"tool_name": str, "tool_args": dict, "tool_result": str},
    ...
]
final_output = "the agent's final answer"
```

then call either the SDK (`aria.sdk.diagnose`) or POST to `/diagnose`. See [CONTRIBUTING.md](../CONTRIBUTING.md#adding-a-framework-adapter) — adapter contributions (CrewAI, AutoGen, smolagents, …) are very welcome.
