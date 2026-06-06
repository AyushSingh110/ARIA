from __future__ import annotations

import json
from pathlib import Path

import dspy

from aria.classifiers.failure_classifier import XGBoostFailureClassifier
from aria.config import get_settings
from aria.dspy_programs.diagnostician import DiagnosticProgram, build_lm
from aria.state.schema import ARIAState
from aria.utils.display import console, print_agent_output, print_phase

_COMPILED_PATH = Path("data/compiled/diagnostician.json")
_xgb = XGBoostFailureClassifier()
_xgb.load(_COMPILED_PATH.parent / "xgb_classifier.json")

_dspy_program: DiagnosticProgram | None = None


def _get_dspy_program() -> DiagnosticProgram:
    global _dspy_program
    if _dspy_program is not None:
        return _dspy_program

    s = get_settings()
    lm = build_lm(api_key=s.groq_api_key, model=f"groq/{s.groq_model}")
    dspy.configure(lm=lm)

    program = DiagnosticProgram()
    if _COMPILED_PATH.exists():
        try:
            program.load(_COMPILED_PATH)
            console.print(f"[dim]Diagnostician: loaded compiled DSPy program from {_COMPILED_PATH}[/dim]")
        except Exception as exc:
            console.print(f"[yellow]Diagnostician: compiled program load failed ({exc}), using zero-shot[/yellow]")
    else:
        console.print("[dim]Diagnostician: no compiled program found — running zero-shot[/dim]")

    _dspy_program = program
    return _dspy_program


def _build_trace_summary(trace: list[dict]) -> str:
    if not trace:
        return "No tool calls."
    lines = []
    for e in trace:
        if e["tool_name"] == "__llm__":
            lines.append(f"  turn {e['turn']}: [LLM output] {e['llm_output'][:80]}")
        else:
            lines.append(
                f"  turn {e['turn']}: {e['tool_name']}({json.dumps(e['tool_args'])[:60]}) "
                f"→ {e['tool_result'][:60]}"
            )
    return "\n".join(lines)


def diagnostician_node(state: ARIAState) -> dict:
    s = get_settings()
    print_phase("diagnose")

    critic_scores = state.get("critic_scores")
    observer_flags = state.get("observer_flags", [])
    trace = state.get("executor_trace", [])

    xgb_prediction = _xgb.predict(dict(state))

    # Build requirement summary for the new Diagnostician input
    req_checklist  = state.get("requirement_checklist") or []
    req_satisfied  = state.get("requirements_satisfied") or []
    req_sat_score  = state.get("requirement_satisfaction", 1.0)
    req_lines = []
    for req, ok in zip(req_checklist, req_satisfied):
        status = "OK" if ok else "MISS"
        req_lines.append(f"REQ: {req} [{status}]")
    requirement_summary = (
        "\n".join(req_lines) if req_lines
        else f"requirement_satisfaction={req_sat_score:.2f} (no checklist available)"
    )

    program = _get_dspy_program()
    try:
        result = program(
            task_description=state["task_description"],
            observer_flags=json.dumps([dict(f) for f in observer_flags], indent=None),
            critic_scores=json.dumps(dict(critic_scores) if critic_scores else {}, indent=None),
            requirement_summary=requirement_summary,
            trace_summary=_build_trace_summary(trace),
        )
        failure_class = result.failure_class.strip().lower()
        manifestation = result.failure_manifestation.strip().lower()
        try:
            confidence = float(result.confidence)
        except (ValueError, TypeError):
            confidence = 0.5
        reasoning = result.reasoning.strip()
    except Exception as exc:
        console.print(f"[red]Diagnostician DSPy error: {exc}[/red]")
        failure_class = "none"
        manifestation = "none"
        confidence = 0.0
        reasoning = f"Diagnosis failed: {exc}"

    if xgb_prediction and xgb_prediction != "none" and failure_class == "none":
        failure_class = xgb_prediction
        confidence = max(confidence, 0.6)
        reasoning = f"[XGBoost signal] {reasoning}"

    # Deterministic overrides — same rules as the API endpoint.
    # The compiled DSPy program (old 4-field signature) can output "none"
    # even when Critic v2 signals a clear failure. These rules fix that.
    obs_flag_types = {f["flag_type"] for f in (state.get("observer_flags") or [])}

    if failure_class == "none" or failure_class is None:
        if req_sat_score < 0.5 and not obs_flag_types:
            failure_class = "goal_misalignment"
            confidence = max(confidence, 0.6)
            reasoning += " [corrected: req_sat<0.5 with no observer flags -> goal_misalignment]"

    if failure_class == "goal_misalignment":
        if "tool_error_loop" in obs_flag_types:
            failure_class = "tool_misuse"
            reasoning += " [corrected: tool_error_loop flag -> tool_misuse]"
        elif "tool_repetition" in obs_flag_types and "tool_error_loop" not in obs_flag_types:
            failure_class = "context_overflow"
            reasoning += " [corrected: tool_repetition, no errors -> context_overflow]"
        elif not obs_flag_types and req_sat_score >= 0.75:
            failure_class = None
            reasoning += " [corrected: no flags + req_sat>=0.75 -> none]"

    if failure_class == "none":
        failure_class = None
        manifestation = None
    elif failure_class is None:
        manifestation = None

    color = "red" if failure_class else "green"
    label = failure_class or "none (clean run)"
    print_agent_output(
        "Diagnostician",
        f"failure_class={label}\n"
        f"manifestation={manifestation or 'none'}\n"
        f"confidence={confidence:.2f}\n"
        f"reasoning: {reasoning[:200]}",
        color=color,
    )

    subtasks = state.get("subtasks", [])
    idx = state.get("current_subtask_index", 0)
    has_more = idx + 1 < len(subtasks)
    next_phase = "decompose" if has_more else "complete"
    next_index = idx + 1 if has_more else idx
    next_subtask = subtasks[next_index] if has_more else state.get("active_subtask")

    return {
        "failure_class": failure_class,
        "failure_manifestation": manifestation,
        "diagnosis_confidence": confidence,
        "diagnosis_reasoning": reasoning,
        "current_phase": next_phase,
        "current_subtask_index": next_index,
        "active_subtask": next_subtask,
        "api_calls_groq": state["api_calls_groq"] + 1,
    }
