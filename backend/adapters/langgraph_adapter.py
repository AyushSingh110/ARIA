"""LangGraph → ARIA adapter.

Converts a LangGraph agent's message history into an ARIA DiagnoseRequest
so any LangGraph agent's trace can be diagnosed without re-running it.

Usage:
    from adapters.langgraph_adapter import langgraph_to_aria
    import requests

    # After your LangGraph agent runs:
    state = graph.invoke({"messages": [...], ...})
    req = langgraph_to_aria(
        messages=state["messages"],
        task_description="The original task you gave the agent",
    )
    # POST to ARIA
    resp = requests.post("http://localhost:8000/diagnose", json=req.model_dump())
    print(resp.json())

Supported message types:
  - HumanMessage    → task input (used for final_output if it appears at end)
  - AIMessage       → LLM response; tool_calls extracted if present
  - ToolMessage     → tool result (matched to preceding AIMessage tool calls)
  - SystemMessage   → skipped

Works with both LangChain message objects and plain dicts (for serialized traces).
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def langgraph_to_aria(
    messages: list[Any],
    task_description: str,
) -> "DiagnoseRequest":  # noqa: F821  (imported below to avoid circular deps)
    """Convert LangGraph message history to an ARIA DiagnoseRequest.

    Args:
        messages: List of LangChain message objects OR plain dicts.
                  Accepted types: AIMessage, ToolMessage, HumanMessage, SystemMessage,
                  or dict with 'type' key ('ai', 'tool', 'human', 'system').
        task_description: The original task given to the agent.

    Returns:
        DiagnoseRequest ready to POST to /diagnose.
    """
    from api.schemas import DiagnoseRequest, ToolCall

    tool_calls: list[ToolCall] = []
    final_output: str = ""
    pending_tool_meta: dict[str, dict] = {}  # tool_call_id → {name, args}
    turn = 0

    for msg in messages:
        msg_type, content, raw_tool_calls, tool_call_id = _parse_message(msg)

        if msg_type == "system":
            continue

        if msg_type == "ai":
            if raw_tool_calls:
                for tc in raw_tool_calls:
                    tc_id   = tc.get("id", f"tc_{turn}")
                    tc_name = tc.get("name", tc.get("function", {}).get("name", "unknown"))
                    tc_args = tc.get("args", tc.get("function", {}).get("arguments", {}))
                    if isinstance(tc_args, str):
                        try:
                            import json
                            tc_args = json.loads(tc_args)
                        except Exception:
                            tc_args = {"raw": tc_args}
                    pending_tool_meta[tc_id] = {"name": tc_name, "args": tc_args}
            else:
                # Plain AI response = final output
                if content:
                    final_output = content

        elif msg_type == "tool":
            meta = pending_tool_meta.pop(tool_call_id, {})
            tool_calls.append(ToolCall(
                tool_name=meta.get("name", "unknown_tool"),
                tool_args=meta.get("args", {}),
                tool_result=content or "",
                turn=turn,
            ))
            turn += 1

        elif msg_type == "human":
            # Last human message after tool exchanges = final answer in some patterns
            pass

    return DiagnoseRequest(
        task_description=task_description,
        tool_calls=tool_calls,
        final_output=final_output,
    )


def _parse_message(msg: Any) -> tuple[str, str, list, str]:
    """Return (type, content, tool_calls, tool_call_id) from any message format."""
    # Dict-style (serialized traces)
    if isinstance(msg, dict):
        msg_type    = msg.get("type", msg.get("role", ""))
        content     = str(msg.get("content", "") or "")
        tool_calls  = msg.get("tool_calls", [])
        tc_id       = msg.get("tool_call_id", "")
        return msg_type, content, tool_calls, tc_id

    # LangChain message objects
    type_name = type(msg).__name__.lower()
    content   = str(getattr(msg, "content", "") or "")
    tc_id     = getattr(msg, "tool_call_id", "")

    if "human" in type_name:
        return "human", content, [], tc_id
    if "system" in type_name:
        return "system", content, [], tc_id
    if "tool" in type_name and "ai" not in type_name:
        return "tool", content, [], tc_id
    if "ai" in type_name or "assistant" in type_name:
        raw_tc = getattr(msg, "tool_calls", []) or []
        return "ai", content, raw_tc, tc_id

    return "unknown", content, [], tc_id


# ── Convenience: diagnose directly ───────────────────────────────────────────

def diagnose_langgraph_trace(
    messages: list[Any],
    task_description: str,
    aria_url: str = "http://localhost:8000",
) -> dict:
    """One-shot: convert LangGraph trace and POST to ARIA /diagnose.

    Returns the ARIA diagnosis dict.
    Requires the ARIA API server to be running.

    Example:
        result = diagnose_langgraph_trace(
            messages=state["messages"],
            task_description="Find the capital of France",
        )
        print(result["failure_class"], result["requirement_satisfaction"])
    """
    import requests

    req = langgraph_to_aria(messages, task_description)
    resp = requests.post(
        f"{aria_url}/diagnose",
        json=req.model_dump(),
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()
