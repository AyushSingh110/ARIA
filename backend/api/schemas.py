from __future__ import annotations
from typing import Optional
from pydantic import BaseModel


class TraceEntry(BaseModel):
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
    """Diagnose a pre-computed trace without running the executor."""
    task_description: str
    trace: list[TraceEntry]
    observer_flags: list[ObserverFlag] = []
    critic_scores: Optional[CriticScores] = None


class RunRequest(BaseModel):
    """Run a task through the full ARIA pipeline and return diagnosis."""
    task: str
    task_class: str = "general"
    max_turns: int = 5


# ── Response models ───────────────────────────────────────────────────────────

class DiagnosisResponse(BaseModel):
    task_id: str
    failure_class: Optional[str]
    confidence: float
    reasoning: str
    manifestation: Optional[str]
    suggested_action: str
    observer_flags: list[dict]
    critic_scores: Optional[dict]
    executor_turn_count: int
    trace_summary: str


class DashboardStats(BaseModel):
    total_runs: int
    class_distribution: dict[str, int]
    class_distribution_pct: dict[str, float]
    avg_confidence: float
    pass_rate: float
    recent_failures: list[dict]
