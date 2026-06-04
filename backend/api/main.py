#!/usr/bin/env python3
"""ARIA Runtime API — FastAPI application.

Endpoints:
  POST /diagnose         — diagnose a pre-computed trace (runs Observer+Critic+Diagnostician)
  POST /diagnose/batch   — diagnose multiple traces at once
  POST /run              — run a task through the full ARIA pipeline
  GET  /dashboard        — distribution and stats over all runs
  GET  /health           — liveness check

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
    BatchDiagnoseRequest,
    BatchDiagnoseResponse,
    DiagnoseRequest,
    DiagnosisResponse,
    DashboardStats,
    FeedbackRequest,
    FeedbackResponse,
    RunRequest,
)

app = FastAPI(
    title="ARIA Runtime API",
    description="Autonomous agent failure detection and diagnosis.",
    version="2.0.0",
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
            lines.append(f"turn {e['turn']}: [LLM] {str(e.get('llm_output', ''))[:80]}")
        else:
            lines.append(
                f"turn {e['turn']}: {e.get('tool_name')}("
                f"{json.dumps(e.get('tool_args', {}))[:40]}) -> "
                f"{str(e.get('tool_result', ''))[:60]}"
            )
    return "\n".join(lines) if lines else "(empty trace)"


def _tool_calls_to_trace(req: DiagnoseRequest) -> list[dict]:
    """Convert DiagnoseRequest.tool_calls into internal trace format."""
    trace = []
    for i, tc in enumerate(req.tool_calls):
        turn = tc.turn if tc.turn >= 0 else i
        trace.append({
            "turn": turn,
            "tool_name": tc.tool_name,
            "tool_args": tc.tool_args,
            "tool_result": tc.tool_result,
            "llm_output": "",
            "latency_ms": 0,
            "token_count": 0,
        })
    # Append final LLM output as last entry if provided
    if req.final_output:
        last_turn = (trace[-1]["turn"] + 1) if trace else 0
        trace.append({
            "turn": last_turn,
            "tool_name": "__llm__",
            "tool_args": {},
            "tool_result": "",
            "llm_output": req.final_output,
            "latency_ms": 0,
            "token_count": 0,
        })
    return trace


def _build_evidence(req_list: list, sat_list: list) -> list[str]:
    """Derive evidence strings from unsatisfied requirements."""
    evidence = []
    for req, sat in zip(req_list, sat_list):
        if not sat:
            evidence.append(f"Requirement not satisfied: '{req}'")
    return evidence


def _post_process_failure_class(
    failure_class: str | None,
    obs_flags: list[dict],
    req_sat: float,
) -> tuple[str | None, str]:
    """Apply deterministic overrides for clear-cut disambiguation cases.

    The LLM Diagnostician can conflate "goal not achieved" with goal_misalignment
    even when the root cause is a tool error. These rules are unambiguous:

    1. tool_error_loop flag present → tool_misuse (tool ran but errored)
    2. tool_repetition flag, no tool errors → context_overflow
    3. LLM returned goal_misalignment but no observer flags, req_sat >= 0.75 → none
    """
    flag_types = {f.get("flag_type") for f in obs_flags}
    override_note = ""

    if "tool_error_loop" in flag_types and failure_class == "goal_misalignment":
        failure_class = "tool_misuse"
        override_note = " [corrected: tool_error_loop flag → tool_misuse]"

    elif "tool_repetition" in flag_types and failure_class == "goal_misalignment" \
            and "tool_error_loop" not in flag_types:
        failure_class = "context_overflow"
        override_note = " [corrected: tool_repetition flag, no errors → context_overflow]"

    elif failure_class == "goal_misalignment" and not obs_flags and req_sat >= 0.75:
        failure_class = None
        override_note = " [corrected: no flags + req_sat≥0.75 → none]"

    return failure_class, override_note


def _signal_based_diagnosis(obs_flags: list[dict], scores: dict) -> tuple[str | None, float]:
    """Fallback diagnosis from observer flags and critic scores when Diagnostician fails."""
    req_sat = scores.get("requirement_satisfaction", 1.0)
    flag_types = [f.get("flag_type") for f in obs_flags]

    # Check requirement satisfaction first (primary signal in v2)
    if req_sat < 0.5:
        return "goal_misalignment", 0.7

    # Observer signals
    if "tool_error_loop" in flag_types:
        return "tool_misuse", 0.65
    if "tool_repetition" in flag_types:
        return "context_overflow", 0.6
    if "prompt_drift" in flag_types:
        return "prompt_drift", 0.6

    if req_sat < 0.75:
        return "goal_misalignment", 0.55

    return None, 0.8


def _run_diagnose(req: DiagnoseRequest) -> DiagnosisResponse:
    """Core diagnosis logic: Observer signals → Critic v2 → Diagnostician."""
    import dspy
    from langchain_core.messages import HumanMessage, SystemMessage

    from aria.config import get_settings
    from aria.dspy_programs.diagnostician import DiagnosticProgram, build_lm
    from langchain_groq import ChatGroq
    from aria.agents.critic import (
        _build_human_message,
        _build_scores,
        _extract_json,
        _SYSTEM_PROMPT,
    )
    from aria.agents.observer import (
        _detect_tool_repetition,
        _detect_tool_error_loop,
    )

    s = get_settings()
    if not s.groq_api_key:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY not configured.")

    # Build trace
    trace = _tool_calls_to_trace(req)

    if not trace and not req.final_output:
        task_id = str(uuid.uuid4())[:8]
        return DiagnosisResponse(
            task_id=task_id,
            failure_class=None,
            confidence=0.0,
            reasoning="Trace is empty — no tool calls or output provided to diagnose.",
            manifestation=None,
            suggested_action="Provide a non-empty trace to diagnose.",
            requirement_satisfaction=0.0,
            requirements=[],
            requirements_satisfied=[],
            evidence=[],
            observer_flags=[],
            critic_scores=None,
            executor_turn_count=0,
            trace_summary="(empty)",
        )

    # Step 1: Observer signals (tool-repetition + error-loop; skip drift/budget
    # since we don't have embeddings or the original max_turns budget here)
    obs_flags: list[dict] = []
    obs_flags.extend([dict(f) for f in _detect_tool_repetition(trace)])
    obs_flags.extend([dict(f) for f in _detect_tool_error_loop(trace)])

    # Step 2: Critic v2 — always use Groq in the API (Groq key already validated above)
    critic_llm = ChatGroq(api_key=s.groq_api_key, model=s.groq_model, temperature=0)
    tool_only_trace = [e for e in trace if e["tool_name"] != "__llm__"]
    human_msg = _build_human_message(req.task_description, req.final_output, tool_only_trace)
    scores: dict = {}
    critic_error: str | None = None
    try:
        response = critic_llm.invoke(
            [SystemMessage(content=_SYSTEM_PROMPT), HumanMessage(content=human_msg)]
        )
        parsed = _extract_json(response.content)
        scores = dict(_build_scores(parsed))
    except Exception as critic_exc:
        critic_error = str(critic_exc)
        scores = {
            "correctness": 2.0, "completeness": 2.0, "efficiency": 3.0, "safety": 5.0,
            "overall": 2.3, "pass_fail": False,
            "requirement_checklist": [], "requirements_satisfied": [], "requirement_satisfaction": 0.0,
            "critic_error": critic_error,
        }

    req_list = scores.get("requirement_checklist", [])
    sat_list = scores.get("requirements_satisfied", [])
    req_sat  = float(scores.get("requirement_satisfaction", 0.0))

    # Build requirement_summary in the format Diagnostician expects:
    # "REQ: <text> [OK/MISS]" per line — lets the model distinguish
    # "tools failed" (tool_misuse) from "tools worked but agent stopped early" (goal_misalignment)
    req_lines = [
        f"REQ: {r} [{'OK' if ok else 'MISS'}]"
        for r, ok in zip(req_list, sat_list)
    ]
    requirement_summary = (
        "\n".join(req_lines) if req_lines
        else f"requirement_satisfaction={req_sat:.2f} (no checklist available)"
    )

    # Step 3: Diagnostician
    trace_summary = _summarise_trace(trace)
    flags_json    = json.dumps(obs_flags)
    scores_json   = json.dumps(scores)

    failure_class: str | None = None
    confidence    = 0.5
    reasoning     = ""
    manifestation: str | None = None

    try:
        lm = build_lm(api_key=s.groq_api_key, model=f"groq/{s.groq_model}")
        dspy.configure(lm=lm)
        program = DiagnosticProgram()
        # Note: we intentionally do NOT load the compiled program here.
        # The compiled program was trained on the old 4-field signature (without
        # requirement_summary). Loading it would override the new disambiguation
        # rules with old few-shot examples that don't understand requirement_summary.
        # Zero-shot with the new signature is more accurate until we re-compile.

        result = program(
            task_description=req.task_description,
            observer_flags=flags_json,
            critic_scores=scores_json,
            requirement_summary=requirement_summary,
            trace_summary=trace_summary,
        )
        raw_class = (getattr(result, "failure_class", "") or "").strip().lower()
        failure_class = None if raw_class == "none" else raw_class
        try:
            confidence = float(getattr(result, "confidence", "0.5"))
        except (ValueError, TypeError):
            confidence = 0.5
        reasoning     = getattr(result, "reasoning", "")
        manifestation = getattr(result, "failure_manifestation", None)
        if manifestation == "none":
            manifestation = None

    except Exception as exc:
        # Fallback: signal-based diagnosis so the endpoint still returns useful info
        failure_class, confidence = _signal_based_diagnosis(obs_flags, scores)
        reasoning = f"Diagnostician unavailable ({exc}). Diagnosis derived from Observer flags and Critic scores."

    # Apply deterministic overrides for clear-cut disambiguation
    failure_class, override_note = _post_process_failure_class(failure_class, obs_flags, req_sat)
    if override_note:
        reasoning = reasoning + override_note

    evidence = _build_evidence(req_list, sat_list)
    if not evidence and failure_class:
        evidence = [f"Behavioral failure detected: {failure_class}"]

    task_id = str(uuid.uuid4())[:8]
    RUN_LOG_DIR.mkdir(parents=True, exist_ok=True)
    # Save full trace so every diagnosis becomes a potential training record
    log = {
        "task_id": task_id,
        "task_description": req.task_description,
        "tool_calls": [{"tool_name": tc.tool_name, "tool_args": tc.tool_args, "tool_result": tc.tool_result} for tc in req.tool_calls],
        "final_output": req.final_output,
        "failure_class": failure_class,
        "confidence": confidence,
        "requirement_satisfaction": req_sat,
        "requirements": req_list,
        "requirements_satisfied": [bool(v) for v in sat_list],
        "evidence": _build_evidence(req_list, sat_list),
        "observer_flags": obs_flags,
        "trace_summary": trace_summary,
        "source": "diagnose_endpoint",
        # human_label and aria_correct added later via /feedback
    }
    (RUN_LOG_DIR / f"{task_id}.json").write_text(json.dumps(log, indent=2), encoding="utf-8")

    return DiagnosisResponse(
        task_id=task_id,
        failure_class=failure_class,
        confidence=confidence,
        reasoning=reasoning,
        manifestation=manifestation,
        suggested_action=_suggested_action(failure_class),
        requirement_satisfaction=req_sat,
        requirements=req_list,
        requirements_satisfied=[bool(v) for v in sat_list],
        evidence=evidence,
        observer_flags=obs_flags,
        critic_scores=scores,
        executor_turn_count=len([e for e in trace if e["tool_name"] != "__llm__"]),
        trace_summary=trace_summary,
    )


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "version": "2.0.0"}


@app.post("/diagnose", response_model=DiagnosisResponse)
def diagnose(req: DiagnoseRequest):
    """Diagnose a pre-computed agent trace.

    Runs Observer signals (tool repetition, error loops) → Critic v2
    (requirement extraction + verification) → Diagnostician.

    Input:
      - task_description: what the agent was asked to do
      - tool_calls: list of {tool_name, tool_args, tool_result}
      - final_output: agent's final response text

    Output includes requirement_satisfaction, per-requirement checklist,
    evidence strings, and failure_class with confidence.
    """
    return _run_diagnose(req)


@app.post("/diagnose/batch", response_model=BatchDiagnoseResponse)
def diagnose_batch(req: BatchDiagnoseRequest):
    """Diagnose a list of pre-computed traces and return all diagnoses.

    Useful for analysing multiple agent runs in one call.
    Each trace is diagnosed independently using the same Observer+Critic+Diagnostician pipeline.
    """
    results = []
    for trace_req in req.traces:
        try:
            result = _run_diagnose(trace_req)
        except HTTPException:
            raise
        except Exception as exc:
            task_id = str(uuid.uuid4())[:8]
            result = DiagnosisResponse(
                task_id=task_id,
                failure_class=None,
                confidence=0.0,
                reasoning=f"Diagnosis error: {exc}",
                manifestation=None,
                suggested_action="Check API logs.",
                requirement_satisfaction=0.0,
                requirements=[],
                requirements_satisfied=[],
                evidence=[str(exc)],
                observer_flags=[],
                critic_scores=None,
                executor_turn_count=0,
                trace_summary="(error)",
            )
        results.append(result)

    dist = Counter(r.failure_class or "none" for r in results)
    return BatchDiagnoseResponse(
        results=results,
        total=len(results),
        failure_distribution=dict(dist),
    )


@app.post("/run", response_model=DiagnosisResponse)
def run_task(req: RunRequest):
    """Run a task through the full ARIA pipeline and return the diagnosis.

    Calls: Orchestrator → Executor → Observer → Critic v2 → Diagnostician.
    Returns the full diagnosis including requirement_satisfaction and evidence.
    """
    from aria.config import get_settings
    from aria.graph import build_graph
    from aria.state import make_initial_state
    import os

    s = get_settings()
    if not s.groq_api_key:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY not configured.")

    initial_state = make_initial_state(
        task_description=req.task,
        task_class=req.task_class,
        max_retries=1,
    )
    os.environ["EXECUTOR_MAX_TURNS"] = str(req.max_turns)

    graph = build_graph()
    try:
        final_state = graph.invoke(initial_state)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {exc}")

    flags         = [dict(f) for f in (final_state.get("observer_flags") or [])]
    scores        = dict(final_state.get("critic_scores") or {})
    trace         = [dict(e) for e in (final_state.get("executor_trace") or [])]
    failure_class = final_state.get("failure_class")
    confidence    = final_state.get("diagnosis_confidence") or 0.0
    reasoning     = final_state.get("diagnosis_reasoning") or ""
    manifestation = final_state.get("failure_manifestation")
    task_id       = final_state.get("task_id", str(uuid.uuid4())[:8])
    req_list      = final_state.get("requirement_checklist") or []
    sat_list      = final_state.get("requirements_satisfied") or []
    req_sat       = float(final_state.get("requirement_satisfaction") or 0.0)

    if manifestation == "none":
        manifestation = None

    evidence = _build_evidence(req_list, sat_list)
    if not evidence and failure_class:
        evidence = [f"Behavioral failure detected: {failure_class}"]

    RUN_LOG_DIR.mkdir(parents=True, exist_ok=True)
    log = {
        "task_id":                  task_id,
        "task_description":         req.task,
        "task":                     req.task,
        "failure_class":            failure_class,
        "confidence":               confidence,
        "requirement_satisfaction": req_sat,
        "requirements":             req_list,
        "requirements_satisfied":   [bool(v) for v in sat_list],
        "evidence":                 evidence,
        "executor_turn_count":      final_state.get("executor_turn_count", 0),
        "source":                   "run_endpoint",
        # human_label and aria_correct added later via /feedback
    }
    (RUN_LOG_DIR / f"{task_id}.json").write_text(json.dumps(log, indent=2), encoding="utf-8")

    return DiagnosisResponse(
        task_id=task_id,
        failure_class=failure_class,
        confidence=confidence,
        reasoning=reasoning,
        manifestation=manifestation,
        suggested_action=_suggested_action(failure_class),
        requirement_satisfaction=req_sat,
        requirements=req_list,
        requirements_satisfied=[bool(v) for v in sat_list],
        evidence=evidence,
        observer_flags=flags,
        critic_scores=scores if scores else None,
        executor_turn_count=final_state.get("executor_turn_count", 0),
        trace_summary=_summarise_trace(trace),
    )


@app.post("/feedback", response_model=FeedbackResponse)
def submit_feedback(req: FeedbackRequest):
    """Submit a human correction for an ARIA diagnosis.

    This is how research-through-usage works: every correction is saved
    alongside the original trace and becomes a labelled training record
    for future Diagnostician re-training.

    - aria_correct=True  → confirms the diagnosis; no human_label needed
    - aria_correct=False → provide human_label with the correct failure class
    """
    run_file = RUN_LOG_DIR / f"{req.task_id}.json"
    if not run_file.exists():
        raise HTTPException(
            status_code=404,
            detail=f"No run found with task_id='{req.task_id}'. "
                   "Only diagnoses made via /diagnose or /run can be labelled.",
        )

    try:
        record = json.loads(run_file.read_text(encoding="utf-8"))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not read run record: {exc}")

    record["aria_correct"]  = req.aria_correct
    record["human_label"]   = record["failure_class"] if req.aria_correct else req.human_label
    record["feedback_notes"] = req.notes

    run_file.write_text(json.dumps(record, indent=2), encoding="utf-8")

    msg = (
        "Diagnosis confirmed — thank you for the label."
        if req.aria_correct
        else f"Correction recorded: ARIA said '{record.get('failure_class')}', human says '{req.human_label}'."
    )
    return FeedbackResponse(task_id=req.task_id, recorded=True, message=msg)


@app.get("/dashboard", response_model=DashboardStats)
def dashboard():
    """Return aggregate statistics over all RealBench and API runs.

    Reads from data/realbench/results/ and data/api_runs/.
    Returns class distribution, average requirement_satisfaction,
    most common failure type, and recent failures list.
    """
    all_results: list[dict] = []

    for f in sorted(RESULTS_DIR.glob("rb_*.json")):
        try:
            r = json.loads(f.read_text(encoding="utf-8"))
            if not r.get("run_error"):
                all_results.append(r)
        except Exception:
            pass

    if RUN_LOG_DIR.exists():
        for f in sorted(RUN_LOG_DIR.glob("*.json")):
            try:
                all_results.append(json.loads(f.read_text(encoding="utf-8")))
            except Exception:
                pass

    if not all_results:
        return DashboardStats(
            total_runs=0,
            class_distribution={},
            class_distribution_pct={},
            avg_confidence=0.0,
            avg_requirement_satisfaction=0.0,
            pass_rate=0.0,
            most_common_failure=None,
            labeled_runs=0,
            human_agreement_rate=None,
            recent_failures=[],
        )

    total    = len(all_results)
    dist     = Counter(
        r.get("aria_label") or r.get("failure_class") or "none"
        for r in all_results
    )
    dist_pct = {k: round(v / total * 100, 1) for k, v in dist.items()}

    confidences = [
        r.get("aria_confidence") or r.get("confidence") or 0.0
        for r in all_results
    ]
    avg_conf = sum(confidences) / len(confidences) if confidences else 0.0

    req_sats = []
    for r in all_results:
        # RealBench results store it under critic_scores
        cs = r.get("critic_scores") or {}
        rs = cs.get("requirement_satisfaction") if isinstance(cs, dict) else None
        if rs is None:
            rs = r.get("requirement_satisfaction")
        if rs is not None:
            try:
                req_sats.append(float(rs))
            except (TypeError, ValueError):
                pass
    avg_req_sat = sum(req_sats) / len(req_sats) if req_sats else 0.0

    pass_count = sum(
        1 for r in all_results
        if (r.get("critic_scores") or {}).get("pass_fail", False)
    )
    pass_rate = pass_count / total if total else 0.0

    # Most common failure (excluding "none")
    failure_counts = {k: v for k, v in dist.items() if k != "none"}
    most_common = max(failure_counts, key=failure_counts.get) if failure_counts else None

    # Human agreement rate — only from runs that have feedback
    # Mentor's key metric: "research through usage" requires tracking this
    labeled = [r for r in all_results if "aria_correct" in r]
    labeled_count = len(labeled)
    if labeled_count > 0:
        agreement_rate = round(sum(1 for r in labeled if r["aria_correct"]) / labeled_count, 3)
    else:
        agreement_rate = None

    recent_failures = [
        {
            "task_id":      r.get("task_id", "?"),
            "task":         (r.get("task") or r.get("task_description", ""))[:60],
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
        avg_requirement_satisfaction=round(avg_req_sat, 3),
        pass_rate=round(pass_rate, 3),
        most_common_failure=most_common,
        labeled_runs=labeled_count,
        human_agreement_rate=agreement_rate,
        recent_failures=recent_failures,
    )
