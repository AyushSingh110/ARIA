#!/usr/bin/env python3
"""ARIA Runtime API — FastAPI application.

Endpoints:
  POST /diagnose   — diagnose a pre-computed trace
  POST /run        — run a task through the full ARIA pipeline
  GET  /dashboard  — distribution and stats over recent runs
  GET  /health     — liveness check

Run:
  cd backend
  uvicorn api.main:app --reload --port 8000

Then open: http://localhost:8000/docs
"""
from __future__ import annotations

import json
import sys
import uuid
from collections import Counter
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from api.schemas import (
    DiagnoseRequest,
    DiagnosisResponse,
    DashboardStats,
    RunRequest,
)

app = FastAPI(
    title="ARIA Runtime API",
    description="Autonomous agent failure detection and diagnosis.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

RESULTS_DIR = Path("data/realbench/results")
RUN_LOG_DIR = Path("data/api_runs")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _suggested_action(failure_class: str | None) -> str:
    return {
        "prompt_drift":       "Re-anchor the agent to the original goal at each turn.",
        "tool_misuse":        "Review tool schemas and argument formats in the system prompt.",
        "context_overflow":   "Inject a running task summary into the system prompt each turn.",
        "goal_misalignment":  "Add explicit success criteria and require the agent to verify them.",
        "hallucination_loop": "Require tool-grounded verification before any factual claim.",
    }.get(failure_class or "", "No action required — run appears clean.")


def _summarise_trace(trace: list[dict]) -> str:
    lines = []
    for e in trace:
        if e.get("tool_name") == "__llm__":
            lines.append(f"turn {e['turn']}: [LLM] {str(e.get('llm_output',''))[:80]}")
        else:
            lines.append(
                f"turn {e['turn']}: {e.get('tool_name')}("
                f"{json.dumps(e.get('tool_args', {}))[:40]}) -> "
                f"{str(e.get('tool_result',''))[:60]}"
            )
    return "\n".join(lines)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "version": "1.0.0"}


@app.post("/diagnose", response_model=DiagnosisResponse)
def diagnose(req: DiagnoseRequest):
    """Diagnose a pre-computed agent trace without re-running the executor.

    Use this when you already have a trace and want ARIA's diagnosis.
    """
    import dspy
    from aria.config import get_settings
    from aria.dspy_programs.diagnostician import DiagnosticProgram, build_lm
    from aria.classifiers.failure_classifier import XGBoostFailureClassifier

    s = get_settings()
    if not s.groq_api_key:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY not configured.")

    lm = build_lm(api_key=s.groq_api_key, model=f"groq/{s.groq_model}")
    dspy.configure(lm=lm)

    program = DiagnosticProgram()
    compiled_path = Path("data/compiled/diagnostician.json")
    if compiled_path.exists():
        try:
            program.load(str(compiled_path))
        except Exception:
            pass

    trace_dicts = [e.model_dump() for e in req.trace]
    trace_summary = _summarise_trace(trace_dicts)
    flags_json    = json.dumps([f.model_dump() for f in req.observer_flags])
    scores_json   = json.dumps(req.critic_scores.model_dump() if req.critic_scores else {})

    try:
        result = program(
            task_description=req.task_description,
            observer_flags=flags_json,
            critic_scores=scores_json,
            trace_summary=trace_summary,
        )
        failure_class = (getattr(result, "failure_class", "") or "").strip().lower()
        if failure_class == "none":
            failure_class = None
        try:
            confidence = float(getattr(result, "confidence", "0.5"))
        except (ValueError, TypeError):
            confidence = 0.5
        reasoning = getattr(result, "reasoning", "")
        manifestation = getattr(result, "failure_manifestation", None)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Diagnostician error: {exc}")

    task_id = str(uuid.uuid4())[:8]

    RUN_LOG_DIR.mkdir(parents=True, exist_ok=True)
    log = {
        "task_id": task_id,
        "task_description": req.task_description,
        "failure_class": failure_class,
        "confidence": confidence,
        "source": "diagnose_endpoint",
    }
    (RUN_LOG_DIR / f"{task_id}.json").write_text(
        json.dumps(log, indent=2), encoding="utf-8"
    )

    return DiagnosisResponse(
        task_id=task_id,
        failure_class=failure_class,
        confidence=confidence,
        reasoning=reasoning,
        manifestation=manifestation,
        suggested_action=_suggested_action(failure_class),
        observer_flags=[f.model_dump() for f in req.observer_flags],
        critic_scores=req.critic_scores.model_dump() if req.critic_scores else None,
        executor_turn_count=len(req.trace),
        trace_summary=trace_summary,
    )


@app.post("/run", response_model=DiagnosisResponse)
def run_task(req: RunRequest):
    """Run a task through the full ARIA pipeline and return the diagnosis.

    This calls Orchestrator -> Executor -> Observer -> Critic -> Diagnostician.
    """
    from aria.config import get_settings
    from aria.graph import build_graph
    from aria.state import make_initial_state

    s = get_settings()
    if not s.groq_api_key:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY not configured.")

    initial_state = make_initial_state(
        task_description=req.task,
        task_class=req.task_class,
        max_retries=1,
    )

    import os
    os.environ["EXECUTOR_MAX_TURNS"] = str(req.max_turns)

    graph = build_graph()
    try:
        final_state = graph.invoke(initial_state)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {exc}")

    flags  = [dict(f) for f in (final_state.get("observer_flags") or [])]
    scores = dict(final_state.get("critic_scores") or {})
    trace  = [dict(e) for e in (final_state.get("executor_trace") or [])]

    failure_class = final_state.get("failure_class")
    confidence    = final_state.get("diagnosis_confidence") or 0.0
    reasoning     = final_state.get("diagnosis_reasoning") or ""
    manifestation = final_state.get("failure_manifestation")
    task_id       = final_state.get("task_id", str(uuid.uuid4())[:8])

    RUN_LOG_DIR.mkdir(parents=True, exist_ok=True)
    log = {
        "task_id":      task_id,
        "task":         req.task,
        "failure_class": failure_class,
        "confidence":   confidence,
        "source":       "run_endpoint",
    }
    (RUN_LOG_DIR / f"{task_id}.json").write_text(
        json.dumps(log, indent=2), encoding="utf-8"
    )

    return DiagnosisResponse(
        task_id=task_id,
        failure_class=failure_class,
        confidence=confidence,
        reasoning=reasoning,
        manifestation=manifestation,
        suggested_action=_suggested_action(failure_class),
        observer_flags=flags,
        critic_scores=scores if scores else None,
        executor_turn_count=final_state.get("executor_turn_count", 0),
        trace_summary=_summarise_trace(trace),
    )


@app.get("/dashboard", response_model=DashboardStats)
def dashboard():
    """Return aggregate statistics over all RealBench and API runs."""
    all_results = []

    for f in sorted(RESULTS_DIR.glob("rb_*.json")):
        r = json.loads(f.read_text(encoding="utf-8"))
        if not r.get("run_error"):
            all_results.append(r)

    for f in sorted(RUN_LOG_DIR.glob("*.json")) if RUN_LOG_DIR.exists() else []:
        all_results.append(json.loads(f.read_text(encoding="utf-8")))

    if not all_results:
        return DashboardStats(
            total_runs=0,
            class_distribution={},
            class_distribution_pct={},
            avg_confidence=0.0,
            pass_rate=0.0,
            recent_failures=[],
        )

    total = len(all_results)
    dist  = Counter(r.get("aria_label") or r.get("failure_class") or "none"
                    for r in all_results)
    dist_pct = {k: round(v / total * 100, 1) for k, v in dist.items()}

    confidences = [r.get("aria_confidence") or r.get("confidence") or 0.0
                   for r in all_results]
    avg_conf = sum(confidences) / len(confidences) if confidences else 0.0

    pass_count = sum(
        1 for r in all_results
        if (r.get("critic_scores") or {}).get("pass_fail", True)
    )
    pass_rate = pass_count / total if total else 0.0

    failures = [
        {
            "task_id":      r.get("task_id", "?"),
            "task":         r.get("task", r.get("task_description", ""))[:60],
            "failure_class": r.get("aria_label") or r.get("failure_class") or "none",
            "confidence":   r.get("aria_confidence") or r.get("confidence") or 0.0,
        }
        for r in all_results
        if (r.get("aria_label") or r.get("failure_class")) not in (None, "none")
    ][-10:]

    return DashboardStats(
        total_runs=total,
        class_distribution=dict(dist),
        class_distribution_pct=dist_pct,
        avg_confidence=round(avg_conf, 3),
        pass_rate=round(pass_rate, 3),
        recent_failures=failures,
    )
