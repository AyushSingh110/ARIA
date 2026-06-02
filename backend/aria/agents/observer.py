from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import cast

from aria.config import get_settings
from aria.memory.embeddings import get_engine
from aria.state.schema import ARIAState, ObserverFlag
from aria.utils.display import (
    console,
    print_agent_output,
    print_observer_flags,
    print_phase,
)


# ── Anomaly detectors ─────────────────────────────────────────────────────────

def _detect_prompt_drift(
    goal_embedding: list[float],
    turn_embeddings: list[list[float]],
    threshold: float,
) -> tuple[list[ObserverFlag], list[float]]:
    """Compute cosine distance from goal embedding for every turn embedding."""
    engine = get_engine(get_settings().embedding_model)
    flags: list[ObserverFlag] = []
    drift_scores: list[float] = []

    for i, emb in enumerate(turn_embeddings):
        dist = engine.cosine_distance(goal_embedding, emb)
        drift_scores.append(dist)
        if dist > threshold:
            flags.append(
                ObserverFlag(
                    flag_type="prompt_drift",
                    signal_value=round(dist, 4),
                    turn=i,
                    description=(
                        f"Turn {i}: cosine distance from goal = {dist:.4f} "
                        f"(threshold {threshold:.2f}). Agent may be drifting."
                    ),
                )
            )
    return flags, drift_scores


def _detect_tool_repetition(trace: list[dict]) -> list[ObserverFlag]:
    """Flag if the same (tool, args_hash) appears more than once."""
    seen: dict[str, int] = {}
    flags: list[ObserverFlag] = []
    for entry in trace:
        tool = entry.get("tool_name", "")
        if tool == "__llm__":
            continue
        key = f"{tool}:{hashlib.md5(json.dumps(entry.get('tool_args', {}), sort_keys=True).encode()).hexdigest()}"
        if key in seen:
            flags.append(
                ObserverFlag(
                    flag_type="tool_repetition",
                    signal_value=0.8,
                    turn=entry.get("turn", -1),
                    description=(
                        f"Turn {entry.get('turn')}: '{tool}' called with identical "
                        f"args as turn {seen[key]}. Possible context overflow."
                    ),
                )
            )
        else:
            seen[key] = entry.get("turn", -1)
    return flags


def _detect_turn_budget(used: int, max_turns: int) -> list[ObserverFlag]:
    """Warn if executor consumed ≥ 80 % of its turn budget."""
    flags: list[ObserverFlag] = []
    ratio = used / max_turns if max_turns else 0.0
    if ratio >= 0.8:
        flags.append(
            ObserverFlag(
                flag_type="turn_budget_warning",
                signal_value=round(ratio, 4),
                turn=used,
                description=(
                    f"Executor used {used}/{max_turns} turns ({ratio*100:.0f}%). "
                    "Agent may be struggling to complete the task."
                ),
            )
        )
    return flags


def _detect_tool_error_loop(trace: list[dict]) -> list[ObserverFlag]:
    """Flag if a tool returns an error and the next call is the same tool."""
    flags: list[ObserverFlag] = []
    for i in range(1, len(trace)):
        prev, curr = trace[i - 1], trace[i]
        if (
            "Error" in prev.get("tool_result", "")
            and prev.get("tool_name") == curr.get("tool_name")
            and prev.get("tool_name") != "__llm__"
        ):
            flags.append(
                ObserverFlag(
                    flag_type="tool_error_loop",
                    signal_value=0.9,
                    turn=curr.get("turn", i),
                    description=(
                        f"Turn {curr.get('turn')}: '{curr.get('tool_name')}' retried "
                        f"after an error on turn {prev.get('turn')}. Possible tool misuse."
                    ),
                )
            )
    return flags


# ── Log writer ────────────────────────────────────────────────────────────────

def _write_observer_log(state: ARIAState, flags: list[ObserverFlag], drift_scores: list[float]) -> str:
    settings = get_settings()
    log_dir = Path(settings.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{state['task_id']}.json"

    log_record = {
        "schema_version": "1.0",
        "task_id": state["task_id"],
        "task_description": state["task_description"],
        "task_class": state["task_class"],
        "run_start_time": state["run_start_time"],
        "run_end_time": datetime.now(timezone.utc).isoformat(),
        "executor_trace": state["executor_trace"],
        "executor_turn_count": state["executor_turn_count"],
        "executor_output": state["executor_output"],
        "observer_flags": [dict(f) for f in flags],
        "drift_scores": drift_scores,
        "goal_embedding_available": bool(state["task_description_embedding"]),
        "turn_embeddings_captured": len(state["goal_embedding_history"]),
        "anomaly_detected": bool(flags),
        "anomaly_severity": max((f["signal_value"] for f in flags), default=0.0),
        "summary": {
            "total_turns": state["executor_turn_count"],
            "tools_called": [
                e["tool_name"] for e in state["executor_trace"] if e["tool_name"] != "__llm__"
            ],
            "unique_tools": list(
                {e["tool_name"] for e in state["executor_trace"] if e["tool_name"] != "__llm__"}
            ),
            "flag_types": list({f["flag_type"] for f in flags}),
        },
    }

    log_path.write_text(json.dumps(log_record, indent=2), encoding="utf-8")
    return str(log_path)


# ── Node ──────────────────────────────────────────────────────────────────────

def observer_node(state: ARIAState) -> dict:
    settings = get_settings()
    print_phase("observe")

    trace = state["executor_trace"]
    goal_emb = state["task_description_embedding"]
    turn_embs = state["goal_embedding_history"]

    all_flags: list[ObserverFlag] = []
    drift_scores: list[float] = []

    # 1. Prompt drift (requires embeddings to be present)
    if goal_emb and turn_embs:
        drift_flags, drift_scores = _detect_prompt_drift(
            goal_emb, turn_embs, settings.anomaly_drift_threshold
        )
        all_flags.extend(drift_flags)

    # 2. Tool repetition
    all_flags.extend(_detect_tool_repetition(trace))

    # 3. Turn budget
    all_flags.extend(_detect_turn_budget(state["executor_turn_count"], settings.executor_max_turns))

    # 4. Tool error loops
    all_flags.extend(_detect_tool_error_loop(trace))

    anomaly_detected = bool(all_flags)
    anomaly_severity = max((f["signal_value"] for f in all_flags), default=0.0)

    # Write full structured log
    log_path = _write_observer_log(state, all_flags, drift_scores)

    print_agent_output(
        "Observer",
        f"Flags raised: {len(all_flags)} | Anomaly: {anomaly_detected} | Severity: {anomaly_severity:.3f}\n"
        f"Log: {log_path}",
        color="yellow",
    )
    print_observer_flags([dict(f) for f in all_flags])

    return {
        "observer_flags": all_flags,
        "anomaly_detected": anomaly_detected,
        "anomaly_severity": anomaly_severity,
        "drift_scores": drift_scores,
        "observer_log_path": log_path,
        "current_phase": "critique",
    }
