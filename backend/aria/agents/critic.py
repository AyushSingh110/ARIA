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

# ── Critic v2 — Requirement-Aware Evaluation ──────────────────────────────────
#
# Root cause of v1 failure: holistic output scoring gave 5.0/5 to runs that
# never wrote required files, never showed required formulas, never found the
# required number of sources. The Critic was scoring "output presence and
# apparent quality" not "requirement satisfaction."
#
# v2 fix: extract explicit requirements first, verify each one against the
# trace and output, compute requirement_satisfaction = satisfied/total.
# pass_fail is now driven by requirement_satisfaction, not holistic overall.

_SYSTEM_PROMPT = """\
You are ARIA's Requirement-Aware Critic (v2). Your job is to evaluate whether \
an AI agent satisfied ALL explicit requirements of its task.

Work through this in three steps:

STEP 1 — EXTRACT REQUIREMENTS
Read the task carefully. List every explicit requirement as a numbered checklist.
A requirement is anything the agent MUST do or produce.
Split compound requirements: "calculate X and show the formula" = TWO requirements.
Typical requirements: compute a value, search N sources, save to a specific file,
include a specific field, satisfy a stated constraint (e.g. "exactly 3", "must check").
Be specific and concrete. Aim for 1–6 requirements per task.

STEP 2 — VERIFY EACH REQUIREMENT
For each requirement, check the trace and executor output.
A requirement is satisfied (true) ONLY if there is clear evidence in the trace or output.
Absence of evidence = not satisfied.
Examples:
  - "save to results.txt" → satisfied only if write_file called with that filename
  - "show the formula" → satisfied only if formula appears in the output
  - "find 3 sources" → satisfied only if 3 distinct sources are listed

STEP 3 — SCORE
- requirement_satisfaction: satisfied_count / total_count as a decimal (e.g. 0.67)
- correctness: 1–5 factual accuracy of what WAS produced (ignore missing requirements here)
- efficiency: 1–5 minimal unnecessary steps
- safety: 1–5 no harmful or irreversible actions

Respond with ONLY a valid JSON object — no markdown, no text outside the JSON:

{
  "requirements": ["<requirement 1>", "<requirement 2>", ...],
  "satisfied": [true, false, ...],
  "requirement_satisfaction": <0.0-1.0>,
  "correctness": <1-5>,
  "efficiency": <1-5>,
  "safety": <1-5>,
  "reasoning": "<one sentence: what was missing or why it passed>"
}

The "requirements" and "satisfied" arrays MUST have the same length.
"""

# Requirement satisfaction threshold for pass/fail
# 0.75 = at least 3 out of 4 requirements must be met
_REQ_PASS_THRESHOLD = 0.75

# Legacy weights kept for overall score (backward compat with Diagnostician)
_WEIGHTS = {"correctness": 0.40, "completeness": 0.30, "efficiency": 0.20, "safety": 0.10}


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
    req_list  = parsed.get("requirements", [])
    sat_list  = parsed.get("satisfied", [])

    # Ensure arrays are same length — truncate to shorter if mismatch
    min_len   = min(len(req_list), len(sat_list))
    req_list  = req_list[:min_len]
    sat_list  = [bool(v) for v in sat_list[:min_len]]

    if min_len > 0:
        req_sat = sum(sat_list) / min_len
    else:
        req_sat = 1.0  # no requirements extracted = assume satisfied

    try:
        req_sat = float(parsed.get("requirement_satisfaction", req_sat))
    except (ValueError, TypeError):
        pass  # use computed value

    correctness = float(parsed.get("correctness", 3))
    efficiency  = float(parsed.get("efficiency",  3))
    safety      = float(parsed.get("safety",      5))

    # completeness is now derived from requirement_satisfaction (scaled 1–5)
    completeness = round(req_sat * 5, 1)

    overall = round(
        correctness * _WEIGHTS["correctness"]
        + completeness * _WEIGHTS["completeness"]
        + efficiency  * _WEIGHTS["efficiency"]
        + safety      * _WEIGHTS["safety"],
        3,
    )

    # pass_fail driven by requirement satisfaction, with correctness floor
    pass_fail = (req_sat >= _REQ_PASS_THRESHOLD) and (correctness >= 2.0)

    return CriticScores(
        correctness=correctness,
        completeness=completeness,
        efficiency=efficiency,
        safety=safety,
        overall=overall,
        pass_fail=pass_fail,
        requirement_checklist=req_list,
        requirements_satisfied=sat_list,
        requirement_satisfaction=round(req_sat, 3),
    )


def _build_human_message(task: str, output: str, trace: list[dict]) -> str:
    trace_lines = []
    for e in trace:
        if e.get("tool_name") == "__llm__":
            trace_lines.append(f"  turn {e['turn']}: [LLM] {str(e.get('llm_output',''))[:120]}")
        else:
            trace_lines.append(
                f"  turn {e['turn']}: {e.get('tool_name')}("
                f"{json.dumps(e.get('tool_args', {}))[:60]}) -> "
                f"{str(e.get('tool_result',''))[:80]}"
            )
    trace_str = "\n".join(trace_lines) if trace_lines else "  (no tool calls)"

    return (
        f"Task: {task}\n\n"
        f"Execution trace:\n{trace_str}\n\n"
        f"Final executor output:\n{output[:600]}"
    )


def critic_node(state: ARIAState) -> dict:
    s = get_settings()
    print_phase("critique")

    task   = state["task_description"]
    output = state.get("executor_output") or "No output produced."
    trace  = state.get("executor_trace", [])

    llm        = _get_llm()
    human_msg  = _build_human_message(task, output, trace)
    response   = llm.invoke([SystemMessage(content=_SYSTEM_PROMPT), HumanMessage(content=human_msg)])

    groq_delta   = 1 if s.critic_provider == "groq"   else 0
    ollama_delta = 1 if s.critic_provider == "ollama" else 0

    try:
        parsed = _extract_json(response.content)
        scores = _build_scores(parsed)
    except Exception as exc:
        console.print(f"[red]Critic v2 parse error: {exc} — using conservative scores[/red]")
        scores = CriticScores(
            correctness=2.0, completeness=2.0, efficiency=3.0, safety=5.0,
            overall=2.3, pass_fail=False,
            requirement_checklist=[],
            requirements_satisfied=[],
            requirement_satisfaction=0.0,
        )

    icon = "+" if scores["pass_fail"] else "x"
    req_n    = len(scores["requirement_checklist"])
    req_sat  = sum(scores["requirements_satisfied"])
    req_pct  = scores["requirement_satisfaction"]

    print_agent_output(
        "Critic v2",
        f"[{icon}] pass={scores['pass_fail']}  "
        f"req_satisfaction={req_sat}/{req_n} ({req_pct:.0%})  "
        f"overall={scores['overall']:.2f}\n"
        + "\n".join(
            f"  {'OK' if ok else 'MISS'} {req}"
            for req, ok in zip(
                scores["requirement_checklist"],
                scores["requirements_satisfied"],
            )
        ),
        color="magenta",
    )

    return {
        "critic_scores":          scores,
        "requirement_checklist":  scores["requirement_checklist"],
        "requirements_satisfied": scores["requirements_satisfied"],
        "requirement_satisfaction": scores["requirement_satisfaction"],
        "current_phase":          "diagnose",
        "api_calls_groq":         state["api_calls_groq"]   + groq_delta,
        "api_calls_ollama":       state["api_calls_ollama"] + ollama_delta,
    }
