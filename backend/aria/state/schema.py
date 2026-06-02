from __future__ import annotations

import operator
import uuid
from datetime import datetime, timezone
from typing import Annotated, Literal, Optional, TypedDict

from langgraph.graph.message import add_messages


# ── Sub-schemas ──────────────────────────────────────────────────────────────

class ExecutorTraceEntry(TypedDict):
    turn: int
    tool_name: str           # "__llm__" when no tool was called
    tool_args: dict
    tool_result: str
    llm_output: str
    latency_ms: int
    token_count: int


class ObserverFlag(TypedDict):
    flag_type: Literal[
        "prompt_drift",
        "tool_repetition",
        "turn_budget_warning",
        "tool_error_loop",
        "general",
    ]
    signal_value: float      # normalised 0.0–1.0 severity
    turn: int
    description: str


class CriticScores(TypedDict):
    correctness: float       # 1–5
    completeness: float
    efficiency: float
    safety: float
    overall: float
    pass_fail: bool


class RefinementRecord(TypedDict):
    target: Literal["system_prompt", "tool_schema", "memory_strategy"]
    target_agent: str
    original_component: str
    refined_component: str
    diff: str
    semantic_distance: float


# ── Master state ──────────────────────────────────────────────────────────────

class ARIAState(TypedDict):
    # Task identity
    task_id: str
    task_description: str
    task_class: str
    task_description_embedding: list[float]

    # Decomposition
    subtasks: list[dict]
    current_subtask_index: int
    active_subtask: Optional[dict]

    # Executor (Annotated[list, operator.add] = append-only reducer)
    executor_output: Optional[str]
    executor_trace: Annotated[list[ExecutorTraceEntry], operator.add]
    executor_turn_count: int
    goal_embedding_history: list[list[float]]
    drift_scores: list[float]

    # Observer (append-only)
    observer_flags: Annotated[list[ObserverFlag], operator.add]
    anomaly_detected: bool
    anomaly_severity: float        # max flag signal_value for the run
    observer_log_path: Optional[str]

    # ── Phase 2 fields (Critic + Diagnostician) ──────────────────
    critic_scores: Optional[CriticScores]
    failure_class: Optional[str]
    failure_manifestation: Optional[str]
    diagnosis_confidence: Optional[float]
    diagnosis_reasoning: Optional[str]

    # ── Phase 3 fields (Refiner + Validator) ─────────────────────
    refinement: Optional[RefinementRecord]
    refinement_applied: bool
    post_refinement_scores: Optional[CriticScores]
    delta_score: Optional[float]
    committed_to_store: bool
    experience_record_id: Optional[str]

    # Control flow
    retry_count: int
    max_retries: int
    escalate: bool
    current_phase: Literal[
        "decompose", "execute", "observe", "critique",
        "diagnose", "refine", "validate", "commit",
        "complete", "escalated",
    ]

    # LangGraph message channel
    messages: Annotated[list, add_messages]

    # Run metadata
    run_start_time: str        # ISO-8601
    total_tokens_used: int
    api_calls_groq: int
    api_calls_ollama: int


# ── Factory ───────────────────────────────────────────────────────────────────

def make_initial_state(
    task_description: str,
    task_class: str = "general",
    max_retries: int = 3,
) -> ARIAState:
    return ARIAState(
        task_id=str(uuid.uuid4()),
        task_description=task_description,
        task_class=task_class,
        task_description_embedding=[],
        subtasks=[],
        current_subtask_index=0,
        active_subtask=None,
        executor_output=None,
        executor_trace=[],
        executor_turn_count=0,
        goal_embedding_history=[],
        drift_scores=[],
        observer_flags=[],
        anomaly_detected=False,
        anomaly_severity=0.0,
        observer_log_path=None,
        critic_scores=None,
        failure_class=None,
        failure_manifestation=None,
        diagnosis_confidence=None,
        diagnosis_reasoning=None,
        refinement=None,
        refinement_applied=False,
        post_refinement_scores=None,
        delta_score=None,
        committed_to_store=False,
        experience_record_id=None,
        retry_count=0,
        max_retries=max_retries,
        escalate=False,
        current_phase="decompose",
        messages=[],
        run_start_time=datetime.now(timezone.utc).isoformat(),
        total_tokens_used=0,
        api_calls_groq=0,
        api_calls_ollama=0,
    )
