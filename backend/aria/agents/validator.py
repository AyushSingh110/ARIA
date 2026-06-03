from __future__ import annotations

from aria.agents.critic import critic_node
from aria.agents.executor import executor_node
from aria.config import get_settings
from aria.state.schema import ARIAState
from aria.store import get_store
from aria.utils.display import console, print_agent_output, print_phase

_COMMIT_THRESHOLD = 0.3


def _apply_refinement(state: ARIAState) -> ARIAState:
    refinement = state.get("refinement")
    if not refinement:
        return state

    patched = dict(state)
    # Reset executor trace so re-run starts fresh
    patched["executor_trace"] = []
    patched["executor_output"] = None
    patched["executor_turn_count"] = 0
    patched["goal_embedding_history"] = []
    patched["refinement_applied"] = True
    return patched


def validator_node(state: ARIAState) -> dict:
    s = get_settings()
    print_phase("validate")

    refinement = state.get("refinement")
    if not refinement:
        console.print("[red]Validator: no refinement to validate — skipping.[/red]")
        return {"current_phase": "complete", "escalate": True}

    original_overall = (state.get("critic_scores") or {}).get("overall", 0.0)

    # Re-run executor with refined prompt injected
    patched_state = _apply_refinement(state)
    executor_result = executor_node(patched_state)

    merged = {**patched_state, **executor_result}
    critic_result = critic_node(merged)

    refined_scores = critic_result.get("critic_scores", {})
    refined_overall = refined_scores.get("overall", 0.0)
    delta = round(refined_overall - original_overall, 3)

    console.print(
        f"  original={original_overall:.2f}  refined={refined_overall:.2f}  delta={delta:+.2f}"
    )

    if delta >= _COMMIT_THRESHOLD:
        record_id = get_store().save({
            "task_id": state["task_id"],
            "task_class": state.get("task_class", "general"),
            "task_description": state["task_description"],
            "failure_class": state.get("failure_class"),
            "failure_manifestation": state.get("failure_manifestation"),
            "refinement_target": refinement["target"],
            "original_component": refinement["original_component"],
            "refined_component": refinement["refined_component"],
            "diff": refinement["diff"],
            "semantic_distance": refinement["semantic_distance"],
            "original_critic_scores": dict(state.get("critic_scores") or {}),
            "refined_critic_scores": dict(refined_scores),
            "delta_score": delta,
            "committed": True,
            "retry_count": state.get("retry_count", 0),
        })
        print_agent_output(
            "Validator",
            f"✓ Committed  delta={delta:+.2f}  record_id={record_id[:16]}…\n"
            f"Experience store: {get_store().count()} records",
            color="bright_green",
        )
        return {
            "post_refinement_scores": refined_scores,
            "delta_score": delta,
            "committed_to_store": True,
            "experience_record_id": record_id,
            "current_phase": "complete",
            "api_calls_groq": state["api_calls_groq"] + critic_result.get("api_calls_groq", 0),
            "api_calls_ollama": state["api_calls_ollama"] + critic_result.get("api_calls_ollama", 0),
        }

    retry = state.get("retry_count", 0) + 1
    max_r = state.get("max_retries", 3)

    # Save uncommitted attempt for future retrieval (still useful signal)
    get_store().save({
        "task_id": state["task_id"],
        "task_class": state.get("task_class", "general"),
        "task_description": state["task_description"],
        "failure_class": state.get("failure_class"),
        "refinement_target": refinement["target"],
        "original_component": refinement["original_component"],
        "refined_component": refinement["refined_component"],
        "delta_score": delta,
        "committed": False,
        "retry_count": state.get("retry_count", 0),
    })

    if retry >= max_r:
        print_agent_output(
            "Validator",
            f"✗ Max retries ({max_r}) reached — escalating.  delta={delta:+.2f}",
            color="bright_red",
        )
        return {
            "post_refinement_scores": refined_scores,
            "delta_score": delta,
            "committed_to_store": False,
            "retry_count": retry,
            "escalate": True,
            "current_phase": "escalated",
        }

    print_agent_output(
        "Validator",
        f"✗ No improvement  delta={delta:+.2f}  retry {retry}/{max_r} — sending back to Refiner",
        color="yellow",
    )
    return {
        "post_refinement_scores": refined_scores,
        "delta_score": delta,
        "committed_to_store": False,
        "retry_count": retry,
        "current_phase": "refine",
        "api_calls_groq": state["api_calls_groq"] + critic_result.get("api_calls_groq", 0),
        "api_calls_ollama": state["api_calls_ollama"] + critic_result.get("api_calls_ollama", 0),
    }
