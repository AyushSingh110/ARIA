from __future__ import annotations

import difflib
import json
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama

from aria.agents.executor import _SYSTEM_PROMPT as EXECUTOR_DEFAULT_PROMPT
from aria.config import get_settings
from aria.memory.embeddings import get_engine
from aria.state.schema import ARIAState, RefinementRecord
from aria.store import get_store
from aria.utils.display import console, print_agent_output, print_phase

_SYSTEM_PROMPT = """\
You are the Refiner agent in ARIA. You fix broken agent behavior by rewriting the specific component that caused the failure.

Rules:
- Make the minimal change needed to fix the root cause
- Keep the rewritten component in the same format and length as the original
- Do not remove safety instructions or resource limits
- Do not change the fundamental purpose of the component
- Output ONLY the rewritten component — no explanation, no markdown fences
"""

_REFINE_GUIDANCE: dict[str, str] = {
    "prompt_drift": (
        "The agent drifted away from the original task goal over turns. "
        "Add explicit goal-anchoring instructions: the agent must re-read the original task "
        "at every turn and confirm its current action is still aligned with it."
    ),
    "tool_misuse": (
        "The agent called the wrong tool, passed wrong arguments, or called tools in the wrong order. "
        "Rewrite the tool usage section to be explicit: which tool to use for which situation, "
        "required argument format, and the correct call sequence."
    ),
    "context_overflow": (
        "The agent lost track of earlier state, repeated completed steps, or violated earlier constraints. "
        "Add explicit context tracking instructions: maintain a running list of what has been done, "
        "never repeat a step already completed, and re-read any constraint stated in turn 0 before acting."
    ),
    "goal_misalignment": (
        "The agent optimised a proxy metric instead of the real objective. "
        "Rewrite to add explicit success criteria that cannot be shortcut: the agent must verify "
        "its output against the original task specification before claiming completion."
    ),
    "hallucination_loop": (
        "The agent repeated confident false information across multiple turns. "
        "Add a mandatory verification rule: any factual claim must be confirmed with a tool call "
        "before being stated. If verification fails, the agent must explicitly say 'unverified'."
    ),
}

_COMMIT_THRESHOLD = 0.3
_MAX_SEMANTIC_DRIFT = 0.65


def _get_llm() -> Any:
    s = get_settings()
    return ChatGroq(api_key=s.groq_api_key, model=s.groq_model, temperature=0.2)


def _build_few_shot_block(examples: list[dict]) -> str:
    if not examples:
        return ""
    lines = ["### Successful refinements for similar failures:\n"]
    for i, ex in enumerate(examples, 1):
        lines.append(f"Example {i} (delta={ex.get('delta_score', '?'):.2f}):")
        lines.append(f"Original:\n{ex.get('original_component', '')[:300]}")
        lines.append(f"Refined:\n{ex.get('refined_component', '')[:300]}\n")
    return "\n".join(lines)


def _pick_target(failure_class: str) -> str:
    return {
        "prompt_drift": "system_prompt",
        "tool_misuse": "tool_schema",
        "context_overflow": "system_prompt",
        "goal_misalignment": "system_prompt",
        "hallucination_loop": "system_prompt",
    }.get(failure_class, "system_prompt")


def _get_original_component(state: ARIAState, target: str) -> str:
    if target in ("system_prompt", "tool_schema"):
        return EXECUTOR_DEFAULT_PROMPT
    return state.get("active_subtask", {}).get("success_criteria", "Task complete.")


def refiner_node(state: ARIAState) -> dict:
    s = get_settings()
    print_phase("refine")

    failure_class = state.get("failure_class") or "prompt_drift"
    task_class = state.get("task_class", "general")
    target = _pick_target(failure_class)
    original = _get_original_component(state, target)
    guidance = _REFINE_GUIDANCE.get(failure_class, _REFINE_GUIDANCE["prompt_drift"])

    similar = get_store().retrieve_similar(failure_class, task_class, k=3)
    few_shot = _build_few_shot_block(similar)

    human_content = (
        f"Task: {state['task_description']}\n\n"
        f"Failure class: {failure_class}\n"
        f"What went wrong: {state.get('diagnosis_reasoning', '')[:300]}\n\n"
        f"Refinement guidance: {guidance}\n\n"
        f"{few_shot}"
        f"Original component to rewrite:\n{original}\n\n"
        f"Write the improved version now:"
    )

    llm = _get_llm()
    response = llm.invoke([SystemMessage(content=_SYSTEM_PROMPT), HumanMessage(content=human_content)])
    refined = response.content.strip()

    engine = get_engine(s.embedding_model)
    sem_distance = engine.cosine_distance(engine.embed(original), engine.embed(refined))

    if sem_distance > _MAX_SEMANTIC_DRIFT:
        console.print(
            f"[yellow]Refiner: semantic drift too high ({sem_distance:.3f} > {_MAX_SEMANTIC_DRIFT})"
            f" — clamping to minimal edit[/yellow]"
        )
        refined = original + f"\n\n[ARIA CORRECTION — {failure_class}]: {guidance}"
        sem_distance = engine.cosine_distance(engine.embed(original), engine.embed(refined))

    diff = "\n".join(difflib.unified_diff(
        original.splitlines(), refined.splitlines(),
        fromfile="original", tofile="refined", lineterm=""
    ))

    record = RefinementRecord(
        target=target,
        target_agent="executor",
        original_component=original,
        refined_component=refined,
        diff=diff,
        semantic_distance=round(sem_distance, 4),
    )

    print_agent_output(
        "Refiner",
        f"target={target}  failure={failure_class}  drift={sem_distance:.3f}\n"
        f"few-shot examples retrieved: {len(similar)}\n"
        f"diff lines: {len(diff.splitlines())}",
        color="blue",
    )

    return {
        "refinement": record,
        "refinement_applied": False,
        "current_phase": "validate",
        "api_calls_groq": state["api_calls_groq"] + 1,
    }
