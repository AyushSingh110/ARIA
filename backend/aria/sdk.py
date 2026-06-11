"""ARIA SDK — three-line integration for diagnosing agent traces.

Local (in-process, needs GROQ_API_KEY in env):

    from aria.sdk import diagnose
    report = diagnose(
        task="Find the population of France and save it to a file",
        tool_calls=[{"tool_name": "web_search",
                     "tool_args": {"query": "population of France"},
                     "tool_result": "67.8 million ..."}],
        final_output="The population of France is 67.8 million.",
    )
    print(report["failure_class"], report["requirement_satisfaction"])

Remote (against a running ARIA API):

    from aria.sdk import diagnose_remote
    report = diagnose_remote(task=..., tool_calls=..., final_output=...,
                             aria_url="http://localhost:8000")

Framework adapters:

    from adapters.langgraph_adapter import diagnose_langgraph_trace
    from adapters.openai_adapter import diagnose_openai_trace
"""
from __future__ import annotations

from typing import Any


def diagnose(
    task: str,
    tool_calls: list[dict] | None = None,
    final_output: str = "",
) -> dict[str, Any]:
    """Diagnose an agent trace in-process (no API server needed).

    Args:
        task: The task the agent was asked to do.
        tool_calls: List of {tool_name, tool_args, tool_result, turn?} dicts.
        final_output: The agent's final answer text.

    Returns:
        Diagnosis dict: failure_class, confidence, reasoning,
        requirement_satisfaction, requirements, evidence, suggested_action.
    """
    from api.main import _run_diagnose
    from api.schemas import DiagnoseRequest, ToolCall

    req = DiagnoseRequest(
        task_description=task,
        tool_calls=[
            ToolCall(
                tool_name=tc.get("tool_name", "unknown"),
                tool_args=tc.get("tool_args", {}),
                tool_result=str(tc.get("tool_result", "")),
                turn=tc.get("turn", i),
            )
            for i, tc in enumerate(tool_calls or [])
        ],
        final_output=final_output,
    )
    return _run_diagnose(req).model_dump()


def diagnose_remote(
    task: str,
    tool_calls: list[dict] | None = None,
    final_output: str = "",
    aria_url: str = "http://localhost:8000",
    timeout: float = 120.0,
) -> dict[str, Any]:
    """Diagnose an agent trace via a running ARIA API server."""
    import httpx

    payload = {
        "task_description": task,
        "tool_calls": tool_calls or [],
        "final_output": final_output,
    }
    resp = httpx.post(f"{aria_url}/diagnose", json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def run_task(
    task: str,
    task_class: str = "general",
) -> dict[str, Any]:
    """Run a task through the full ARIA pipeline (Orchestrator → ... → Diagnostician).

    Requires GROQ_API_KEY (and Ollama if configured as executor/critic provider).
    """
    from aria.graph import build_graph
    from aria.state import make_initial_state

    state = make_initial_state(task_description=task, task_class=task_class, max_retries=1)
    final = build_graph().invoke(state)
    return {
        "failure_class": final.get("failure_class"),
        "confidence": final.get("diagnosis_confidence"),
        "reasoning": final.get("diagnosis_reasoning"),
        "requirement_satisfaction": final.get("requirement_satisfaction"),
        "requirements": final.get("requirement_checklist"),
        "requirements_satisfied": final.get("requirements_satisfied"),
        "output": final.get("executor_output"),
        "grounding": final.get("grounding"),
    }
