#!/usr/bin/env python3
"""ARIA-RealBench v1 — Run real tasks through the full ARIA pipeline.

Reads tasks from data/realbench/tasks.json
Saves each result to data/realbench/results/rb_NNN.json
Supports resume: skips tasks already completed.

Run all:   python scripts/run_realbench.py
Run one:   python scripts/run_realbench.py --id rb_006
Run range: python scripts/run_realbench.py --start 1 --end 10
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import click

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

TASKS_PATH   = Path("data/realbench/tasks.json")
RESULTS_DIR  = Path("data/realbench/results")


def load_tasks(start: int | None, end: int | None, task_id: str | None) -> list[dict]:
    if not TASKS_PATH.exists():
        print(f"ERROR: {TASKS_PATH} not found.")
        sys.exit(1)
    all_tasks = json.loads(TASKS_PATH.read_text(encoding="utf-8"))["tasks"]
    if task_id:
        tasks = [t for t in all_tasks if t["id"] == task_id]
        if not tasks:
            print(f"ERROR: task id '{task_id}' not found.")
            sys.exit(1)
        return tasks
    if start is not None or end is not None:
        s = (start or 1) - 1
        e = end or len(all_tasks)
        return all_tasks[s:e]
    return all_tasks


def already_done(task_id: str) -> bool:
    return (RESULTS_DIR / f"{task_id}.json").exists()


def run_task(task: dict) -> dict:
    from aria.config import get_settings
    from aria.graph import build_graph
    from aria.state import make_initial_state

    settings = get_settings()
    initial_state = make_initial_state(
        task_description=task["task"],
        task_class=task["task_class"],
        max_retries=1,
    )

    graph = build_graph()
    try:
        final_state = graph.invoke(initial_state)
        success = True
        error = None
    except Exception as exc:
        final_state = initial_state
        success = False
        error = str(exc)

    # Extract key fields for the result record
    flags  = [dict(f) for f in (final_state.get("observer_flags") or [])]
    scores = dict(final_state.get("critic_scores") or {})
    trace  = [dict(e) for e in (final_state.get("executor_trace") or [])]

    result = {
        "task_id":            task["id"],
        "task":               task["task"],
        "task_class":         task["task_class"],
        "expected_class":     task.get("expected_class", "unknown"),
        "task_notes":         task.get("notes", ""),

        # ARIA diagnosis
        "aria_label":         final_state.get("failure_class"),
        "aria_confidence":    final_state.get("diagnosis_confidence"),
        "aria_reasoning":     (final_state.get("diagnosis_reasoning") or "")[:400],
        "aria_manifestation": final_state.get("failure_manifestation"),

        # Observer signals
        "observer_flags":     flags,
        "anomaly_detected":   final_state.get("anomaly_detected", False),
        "anomaly_severity":   final_state.get("anomaly_severity", 0.0),
        "drift_scores":       final_state.get("drift_scores") or [],

        # Executor
        "executor_output":    (final_state.get("executor_output") or "")[:500],
        "executor_turn_count": final_state.get("executor_turn_count", 0),
        "trace_summary":      _summarise(trace),

        # Critic
        "critic_scores":      scores,

        # Human review (filled by review_realbench.py)
        "human_label":        None,
        "human_notes":        None,
        "reviewed":           False,

        # Run metadata
        "run_success":        success,
        "run_error":          error,
        "run_timestamp":      __import__("datetime").datetime.utcnow().isoformat(),
    }
    return result


def _summarise(trace: list[dict]) -> str:
    lines = []
    for e in trace:
        if e.get("tool_name") == "__llm__":
            lines.append(f"turn {e['turn']}: [LLM] {str(e.get('llm_output',''))[:80]}")
        else:
            lines.append(
                f"turn {e['turn']}: {e.get('tool_name')}("
                f"{json.dumps(e.get('tool_args',{}))[:40]}) -> "
                f"{str(e.get('tool_result',''))[:60]}"
            )
    return "\n".join(lines)


def _preflight():
    """Check providers are reachable before running tasks."""
    from aria.config import get_settings
    import httpx

    s = get_settings()

    if not s.groq_api_key:
        print("ERROR: GROQ_API_KEY not set in .env")
        sys.exit(1)

    ollama_needed = any(
        p == "ollama" for p in [s.orchestrator_provider, s.executor_provider, s.critic_provider]
    )
    if ollama_needed:
        try:
            httpx.get(f"{s.ollama_base_url}/api/tags", timeout=3.0)
        except Exception:
            print(f"\nERROR: Ollama not reachable at {s.ollama_base_url}")
            print("One or more providers are set to 'ollama' but Ollama is not running.")
            print("\nFix: add these lines to your .env file:")
            print("  EXECUTOR_PROVIDER=groq")
            print("  CRITIC_PROVIDER=groq")
            print("  EXECUTOR_MAX_TURNS=5")
            sys.exit(1)


@click.command()
@click.option("--id",    "task_id", default=None, help="Run single task by ID (e.g. rb_006)")
@click.option("--start", default=None, type=int,  help="Start task number (1-based)")
@click.option("--end",   default=None, type=int,  help="End task number (inclusive)")
@click.option("--force", is_flag=True, help="Re-run tasks even if already completed")
@click.option("--delay", default=8, type=float, show_default=True,
              help="Seconds to wait between tasks (avoids Groq rate limits)")
def main(task_id, start, end, force, delay):
    """Run ARIA-RealBench tasks through the full ARIA pipeline."""
    _preflight()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    tasks = load_tasks(start, end, task_id)

    skipped = 0
    ran = 0
    errors = 0

    print(f"\nARIA-RealBench v1 — {len(tasks)} tasks queued")
    print(f"Results -> {RESULTS_DIR}/\n")
    print(f"{'ID':<10} {'Expected':<20} {'ARIA Label':<20} {'Conf':>5}  Status")
    print("-" * 70)

    for i, task in enumerate(tasks):
        if not force and already_done(task["id"]):
            skipped += 1
            continue

        result = run_task(task)
        out_path = RESULTS_DIR / f"{task['id']}.json"
        out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

        label = result["aria_label"] or "none"
        conf  = result["aria_confidence"] or 0.0
        status = "ERROR" if result["run_error"] else "OK"

        if result["run_error"]:
            errors += 1
            print(
                f"{task['id']:<10} "
                f"{task.get('expected_class','?'):<20} "
                f"{'ERROR':<20} "
                f"{'':>4}  {result['run_error'][:60]}"
            )
        else:
            ran += 1
            print(
                f"{task['id']:<10} "
                f"{task.get('expected_class','?'):<20} "
                f"{label:<20} "
                f"{conf:>4.2f}  OK"
            )

        if i < len(tasks) - 1 and not result["run_error"]:
            time.sleep(delay)

    print("-" * 70)
    print(f"Done: {ran} ran  |  {skipped} skipped  |  {errors} errors")
    print(f"\nNext: python scripts/review_realbench.py")


if __name__ == "__main__":
    main()
