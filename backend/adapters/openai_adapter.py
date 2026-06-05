"""OpenAI → ARIA adapter.

Converts OpenAI Assistants API run steps OR chat completions with tool_calls
into an ARIA DiagnoseRequest so any OpenAI agent trace can be diagnosed.

Supported input formats:

  1. OpenAI Assistants API run steps list:
       steps = client.beta.threads.runs.steps.list(thread_id=..., run_id=...)
       req = openai_to_aria(steps.data, task_description="...")

  2. OpenAI chat completions message list (with tool_calls):
       messages = [{"role": "user", ...}, {"role": "assistant", "tool_calls": [...]}, ...]
       req = openai_to_aria(messages, task_description="...", format="chat")

Usage:
    from adapters.openai_adapter import openai_to_aria, diagnose_openai_trace
    import requests

    req = openai_to_aria(run_steps, task_description="Your task here")
    resp = requests.post("http://localhost:8000/diagnose", json=req.model_dump())
    print(resp.json())
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Literal

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def openai_to_aria(
    data: list[Any],
    task_description: str,
    format: Literal["run_steps", "chat"] = "run_steps",
) -> "DiagnoseRequest":  # noqa: F821
    """Convert an OpenAI agent trace to an ARIA DiagnoseRequest.

    Args:
        data: List of OpenAI run step objects OR chat message dicts.
        task_description: The original task given to the agent.
        format: "run_steps" for Assistants API, "chat" for chat completions.

    Returns:
        DiagnoseRequest ready to POST to /diagnose.
    """
    if format == "chat":
        return _from_chat_messages(data, task_description)
    return _from_run_steps(data, task_description)


def _from_run_steps(steps: list[Any], task_description: str) -> "DiagnoseRequest":
    """Parse OpenAI Assistants API run steps."""
    from api.schemas import DiagnoseRequest, ToolCall

    tool_calls: list[ToolCall] = []
    final_output: str = ""

    for turn, step in enumerate(reversed(steps)):  # steps are newest-first
        step_type = _get(step, "type")

        if step_type == "tool_calls":
            for tc in (_get(step, "step_details", "tool_calls") or []):
                fn   = _get(tc, "function") or {}
                name = _get(tc, "type") or "function"  # code_interpreter / retrieval / function
                if name == "function":
                    name = _get(fn, "name") or "function"

                args_raw = _get(fn, "arguments") or "{}"
                try:
                    args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
                except Exception:
                    args = {"raw": str(args_raw)}

                output = _get(fn, "output") or _get(tc, "code_interpreter", "outputs") or ""
                if isinstance(output, list):
                    output = " | ".join(str(o) for o in output)

                tool_calls.append(ToolCall(
                    tool_name=name,
                    tool_args=args if isinstance(args, dict) else {},
                    tool_result=str(output),
                    turn=turn,
                ))

        elif step_type == "message_creation":
            msg_id = _get(step, "step_details", "message_creation", "message_id")
            if msg_id:
                final_output = f"[message_id: {msg_id}]"

    from api.schemas import DiagnoseRequest
    return DiagnoseRequest(
        task_description=task_description,
        tool_calls=tool_calls,
        final_output=final_output,
    )


def _from_chat_messages(messages: list[Any], task_description: str) -> "DiagnoseRequest":
    """Parse OpenAI chat completions message list."""
    from api.schemas import DiagnoseRequest, ToolCall

    tool_calls: list[ToolCall] = []
    final_output: str = ""
    pending: dict[str, dict] = {}  # tool_call_id → {name, args}
    turn = 0

    for msg in messages:
        role    = _get(msg, "role") or ""
        content = str(_get(msg, "content") or "")

        if role == "assistant":
            raw_tcs = _get(msg, "tool_calls") or []
            if raw_tcs:
                for tc in raw_tcs:
                    tc_id = _get(tc, "id") or f"tc_{turn}"
                    fn    = _get(tc, "function") or {}
                    name  = _get(fn, "name") or "function"
                    args_raw = _get(fn, "arguments") or "{}"
                    try:
                        args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
                    except Exception:
                        args = {"raw": str(args_raw)}
                    pending[tc_id] = {"name": name, "args": args}
            elif content:
                final_output = content

        elif role == "tool":
            tc_id = _get(msg, "tool_call_id") or ""
            meta  = pending.pop(tc_id, {})
            tool_calls.append(ToolCall(
                tool_name=meta.get("name", "unknown"),
                tool_args=meta.get("args", {}),
                tool_result=content,
                turn=turn,
            ))
            turn += 1

    return DiagnoseRequest(
        task_description=task_description,
        tool_calls=tool_calls,
        final_output=final_output,
    )


def _get(obj: Any, *keys: str) -> Any:
    """Safe nested attribute/key access for both objects and dicts."""
    cur = obj
    for k in keys:
        if cur is None:
            return None
        if isinstance(cur, dict):
            cur = cur.get(k)
        else:
            cur = getattr(cur, k, None)
    return cur


# ── Convenience: diagnose directly ───────────────────────────────────────────

def diagnose_openai_trace(
    data: list[Any],
    task_description: str,
    format: Literal["run_steps", "chat"] = "run_steps",
    aria_url: str = "http://localhost:8000",
) -> dict:
    """One-shot: convert OpenAI trace and POST to ARIA /diagnose.

    Returns the ARIA diagnosis dict.
    Requires the ARIA API server to be running.
    """
    import requests

    req = openai_to_aria(data, task_description, format=format)
    resp = requests.post(
        f"{aria_url}/diagnose",
        json=req.model_dump(),
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()
