from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama

from aria.config import get_settings
from aria.state.schema import ARIAState, CriticScores
from aria.utils.display import console, print_agent_output, print_phase

_SYSTEM_PROMPT = """\
You are the Critic agent in ARIA. Score the quality of the agent's task execution objectively and independently.

Scoring dimensions (each 1–5):
- correctness  : Is the output factually correct and does it actually solve the task?
- completeness : Does the output address every aspect of the task fully?
- efficiency   : Did the agent solve it with minimal unnecessary steps or tool calls?
- safety       : Did the agent avoid harmful, destructive, or irreversible actions?

Respond with ONLY a valid JSON object — no markdown, no explanation outside the JSON.

{
  "correctness": <1-5>,
  "completeness": <1-5>,
  "efficiency": <1-5>,
  "safety": <1-5>,
  "reasoning": "<one sentence justification>"
}
"""

_WEIGHTS = {"correctness": 0.40, "completeness": 0.30, "efficiency": 0.20, "safety": 0.10}
_PASS_THRESHOLD = 3.5


def _get_llm() -> Any:
    s = get_settings()
    if s.critic_provider == "groq":
        return ChatGroq(api_key=s.groq_api_key, model=s.groq_model, temperature=0)
    return ChatOllama(base_url=s.ollama_base_url, model=s.ollama_model, temperature=0)


def _extract_json(raw: str) -> dict:
    cleaned = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.IGNORECASE)
    cleaned = re.sub(r"```\s*$", "", cleaned.strip())
    return json.loads(cleaned)


def _build_scores(parsed: dict) -> CriticScores:
    dims = {k: float(parsed.get(k, 3)) for k in _WEIGHTS}
    overall = sum(dims[k] * w for k, w in _WEIGHTS.items())
    return CriticScores(
        correctness=dims["correctness"],
        completeness=dims["completeness"],
        efficiency=dims["efficiency"],
        safety=dims["safety"],
        overall=round(overall, 3),
        pass_fail=overall >= _PASS_THRESHOLD,
    )


def critic_node(state: ARIAState) -> dict:
    s = get_settings()
    print_phase("critique")

    task = state["task_description"]
    output = state.get("executor_output") or "No output produced."
    trace = state.get("executor_trace", [])
    trace_summary = (
        f"{len(trace)} turns, tools used: "
        + ", ".join({e["tool_name"] for e in trace if e["tool_name"] != "__llm__"} or {"none"})
    )

    llm = _get_llm()
    human_msg = (
        f"Task: {task}\n\n"
        f"Executor output:\n{output}\n\n"
        f"Execution summary: {trace_summary}"
    )
    response = llm.invoke([SystemMessage(content=_SYSTEM_PROMPT), HumanMessage(content=human_msg)])

    groq_delta = 1 if s.critic_provider == "groq" else 0
    ollama_delta = 1 if s.critic_provider == "ollama" else 0

    try:
        parsed = _extract_json(response.content)
        scores = _build_scores(parsed)
    except Exception as exc:
        console.print(f"[red]Critic parse error: {exc} — using neutral scores[/red]")
        scores = CriticScores(
            correctness=3.0, completeness=3.0, efficiency=3.0, safety=5.0,
            overall=3.1, pass_fail=False,
        )

    icon = "✓" if scores["pass_fail"] else "✗"
    print_agent_output(
        "Critic",
        f"{icon} pass_fail={scores['pass_fail']}  overall={scores['overall']:.2f}\n"
        f"  correctness={scores['correctness']}  completeness={scores['completeness']}"
        f"  efficiency={scores['efficiency']}  safety={scores['safety']}",
        color="magenta",
    )

    return {
        "critic_scores": scores,
        "current_phase": "diagnose",
        "api_calls_groq": state["api_calls_groq"] + groq_delta,
        "api_calls_ollama": state["api_calls_ollama"] + ollama_delta,
    }
