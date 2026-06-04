from __future__ import annotations
from typing import Optional
from pydantic import BaseModel


class ToolCall(BaseModel):
    """A single tool call in an agent trace."""
    tool_name: str
    tool_args: dict = {}
    tool_result: str = ""
    turn: int = -1      # -1 = auto-assign by position


class TraceEntry(BaseModel):
    """Legacy trace entry format (kept for backward compat)."""
    turn: int
    tool_name: str
    tool_args: dict = {}
    tool_result: str = ""
    llm_output: str = ""
    latency_ms: int = 0
    token_count: int = 0


class ObserverFlag(BaseModel):
    flag_type: str
    signal_value: float
    turn: int
    description: str


class CriticScores(BaseModel):
    correctness: float
    completeness: float
    efficiency: float
    safety: float
    overall: float
    pass_fail: bool


# ── Request models ────────────────────────────────────────────────────────────

class DiagnoseRequest(BaseModel):
    """Diagnose a pre-computed agent trace without running the executor.

    Natural format: provide task + tool_calls + final_output.
    Observer signals and Critic v2 are run automatically on the trace.
    """
    task_description: str
    tool_calls: list[ToolCall] = []
    final_output: str = ""


class RunRequest(BaseModel):
    """Run a task through the full ARIA pipeline and return diagnosis."""
    task: str
    task_class: str = "general"
    max_turns: int = 5


class BatchDiagnoseRequest(BaseModel):
    """Diagnose a batch of pre-computed traces."""
    traces: list[DiagnoseRequest]


# ── Response models ───────────────────────────────────────────────────────────

class DiagnosisResponse(BaseModel):
    task_id: str
    failure_class: Optional[str]
    confidence: float
    reasoning: str
    manifestation: Optional[str]
    suggested_action: str
    # v2 requirement-aware fields
    requirement_satisfaction: float = 0.0
    requirements: list[str] = []
    requirements_satisfied: list[bool] = []
    evidence: list[str] = []
    # observer / trace metadata
    observer_flags: list[dict] = []
    critic_scores: Optional[dict] = None
    executor_turn_count: int
    trace_summary: str


class BatchDiagnoseResponse(BaseModel):
    results: list[DiagnosisResponse]
    total: int
    failure_distribution: dict[str, int]


class FeedbackRequest(BaseModel):
    """Human correction for an ARIA diagnosis. Saved as training data."""
    task_id: str
    aria_correct: bool                  # was ARIA's diagnosis correct?
    human_label: Optional[str] = None  # correct class if aria_correct=False
    notes: str = ""                     # optional free-text reasoning


class FeedbackResponse(BaseModel):
    task_id: str
    recorded: bool
    message: str


class DashboardStats(BaseModel):
    total_runs: int
    class_distribution: dict[str, int]
    class_distribution_pct: dict[str, float]
    avg_confidence: float
    avg_requirement_satisfaction: float
    pass_rate: float
    most_common_failure: Optional[str]
    labeled_runs: int                       # runs with human feedback
    human_agreement_rate: Optional[float]   # fraction where ARIA was correct
    recent_failures: list[dict]
