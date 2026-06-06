#!/usr/bin/env python3
"""Run GAIA tasks through the full ARIA pipeline in configurable batches.

Designed for batch-by-batch execution so you can swap API keys between
batches if you hit Groq rate limits.

Workflow:
  # Download tasks first (one-time):
  python scripts/gaia_download.py

  # Run batch 0 (tasks 1-5):
  python scripts/gaia_run_batch.py --batch 0

  # [Optional: swap GROQ_API_KEY in .env]

  # Run batch 1 (tasks 6-10):
  python scripts/gaia_run_batch.py --batch 1

  # Run ALL batches automatically (adds delay between batches):
  python scripts/gaia_run_batch.py --all

  # Check progress:
  python scripts/gaia_run_batch.py --status

Results are saved to: data/gaia/results/{gaia_task_id}.json
Progress is tracked in: data/gaia/progress.json
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import click

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

GAIA_DIR     = Path("data/gaia")
TASKS_FILE   = GAIA_DIR / "tasks.json"
RESULTS_DIR  = GAIA_DIR / "results"
PROGRESS_FILE = GAIA_DIR / "progress.json"


# ── Progress tracking ─────────────────────────────────────────────────────────

def _load_progress() -> dict:
    base = {"completed": [], "failed": [], "skipped": []}
    if PROGRESS_FILE.exists():
        try:
            saved = json.loads(PROGRESS_FILE.read_text(encoding="utf-8"))
            base.update(saved)        # merge — adds any missing keys
        except Exception:
            pass
    return base


def _save_progress(progress: dict) -> None:
    PROGRESS_FILE.write_text(json.dumps(progress, indent=2), encoding="utf-8")


def _load_tasks() -> list[dict]:
    if not TASKS_FILE.exists():
        print(f"ERROR: {TASKS_FILE} not found. Run: python scripts/gaia_download.py")
        sys.exit(1)
    data = json.loads(TASKS_FILE.read_text(encoding="utf-8"))
    return data["tasks"]


# ── Preflight check ───────────────────────────────────────────────────────────

def _preflight():
    from aria.config import get_settings
    import httpx

    s = get_settings()

    if not s.groq_api_key:
        print("ERROR: GROQ_API_KEY not set in .env")
        sys.exit(1)

    ollama_needed = any(
        p == "ollama"
        for p in [s.orchestrator_provider, s.executor_provider, s.critic_provider]
    )
    if ollama_needed:
        try:
            httpx.get(f"{s.ollama_base_url}/api/tags", timeout=3.0)
        except Exception:
            print(f"\nERROR: Ollama not reachable at {s.ollama_base_url}")
            print("Providers set to 'ollama' but Ollama is not running.")
            print("\nEither start Ollama, or set in .env:")
            print("  EXECUTOR_PROVIDER=groq")
            print("  CRITIC_PROVIDER=groq")
            sys.exit(1)

    print(f"  Orchestrator : {s.orchestrator_provider} ({s.groq_model})")
    print(f"  Executor     : {s.executor_provider} ({s.ollama_model if s.executor_provider == 'ollama' else s.groq_model})")
    print(f"  Critic       : {s.critic_provider}")
    print(f"  Max turns    : {s.executor_max_turns}")


# ── Answer matching ───────────────────────────────────────────────────────────

def _answer_matches(expected: str, output: str) -> bool:
    """Check if the agent's output contains the expected answer.

    Handles: case normalization, numeric equivalence, abbreviations
    (Saint/St., Mount/Mt.), and word-boundary matching.
    """
    import re

    exp = expected.lower().strip()
    out = output.lower()

    # Direct substring match
    if exp in out:
        return True

    # Abbreviation expansions (Saint Petersburg / St. Petersburg)
    _ABBREV = {"saint": "st", "street": "st", "mount": "mt", "doctor": "dr", "fort": "ft"}
    for full, short in _ABBREV.items():
        if exp.replace(full, short) in out:
            return True
        if exp.replace(short, full) in out:
            return True

    # Numeric: compare as float (handles "6" vs "6.0", "1,234" vs "1234")
    try:
        exp_num = float(exp.replace(",", ""))
        for m in re.findall(r"-?\d[\d,]*\.?\d*", out):
            try:
                if abs(float(m.replace(",", "")) - exp_num) < 0.01:
                    return True
            except ValueError:
                pass
    except ValueError:
        pass

    # Word-boundary match for single-word answers
    if re.search(r"\b" + re.escape(exp) + r"\b", out):
        return True

    return False


# ── Task runner ───────────────────────────────────────────────────────────────

def _run_one(task: dict) -> dict:
    from aria.graph import build_graph
    from aria.state import make_initial_state

    initial_state = make_initial_state(
        task_description=task["task_description"],
        task_class=task["task_class"],
        max_retries=1,
    )
    graph = build_graph()
    try:
        final_state = graph.invoke(initial_state)
        run_error = None
    except Exception as exc:
        final_state = initial_state
        run_error = str(exc)

    flags  = [dict(f) for f in (final_state.get("observer_flags") or [])]
    scores = dict(final_state.get("critic_scores") or {})
    trace  = [dict(e) for e in (final_state.get("executor_trace") or [])]
    output = final_state.get("executor_output") or ""

    # Auto-check answer correctness (fuzzy match)
    expected = task.get("expected_answer", "")
    gaia_correct: bool | None = None
    if expected and output:
        gaia_correct = _answer_matches(expected, output)

    return {
        # GAIA identity
        "gaia_task_id":   task["gaia_task_id"],
        "question":       task["question"],
        "level":          task["level"],
        "expected_answer": expected,
        "task_class":     task["task_class"],

        # ARIA diagnosis
        "aria_label":          final_state.get("failure_class"),
        "aria_confidence":     final_state.get("diagnosis_confidence"),
        "aria_reasoning":      (final_state.get("diagnosis_reasoning") or "")[:400],
        "aria_manifestation":  final_state.get("failure_manifestation"),
        "requirement_satisfaction": final_state.get("requirement_satisfaction", 0.0),
        "requirement_checklist":   final_state.get("requirement_checklist") or [],
        "requirements_satisfied":  final_state.get("requirements_satisfied") or [],

        # Observer
        "observer_flags":    flags,
        "anomaly_detected":  final_state.get("anomaly_detected", False),
        "anomaly_severity":  final_state.get("anomaly_severity", 0.0),
        "drift_scores":      final_state.get("drift_scores") or [],

        # Executor
        "executor_output":    output[:600],
        "executor_turn_count": final_state.get("executor_turn_count", 0),
        "trace":              trace,
        "trace_summary":      _summarise(trace),
        "critic_scores":      scores,

        # GAIA answer check
        "gaia_correct": gaia_correct,

        # Human review fields (filled by manual review later)
        "human_label":  None,
        "human_notes":  None,
        "reviewed":     False,

        # Run metadata
        "run_error":     run_error,
        "run_timestamp": datetime.now(timezone.utc).isoformat(),
    }


def _summarise(trace: list[dict]) -> str:
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
    return "\n".join(lines)


# ── Batch execution ───────────────────────────────────────────────────────────

def _run_batch(tasks: list[dict], batch_num: int, delay: float, force: bool) -> dict:
    progress = _load_progress()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    ran = skipped = errors = 0
    print(f"\nBatch {batch_num} — {len(tasks)} tasks")
    print(f"{'GAIA ID':<38} {'Level':>5} {'ARIA Label':<22} {'ReqSat':>6}  {'GAIA?':>5}  Status")
    print("-" * 88)

    for i, task in enumerate(tasks):
        tid = task["gaia_task_id"]
        out_path = RESULTS_DIR / f"{tid}.json"

        if not force and tid in progress["completed"]:
            skipped += 1
            print(f"{tid:<38} {'—':>5} {'(skipped)':<22}")
            continue

        result = _run_one(task)
        out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

        if result["run_error"]:
            errors += 1
            progress["failed"].append(tid)
            print(
                f"{tid:<38} {task['level']:>5} {'ERROR':<22} {'':>6}  {'':>5}  "
                f"{result['run_error'][:40]}"
            )
        else:
            ran += 1
            if tid in progress.get("failed", []):
                progress["failed"].remove(tid)
            progress["completed"].append(tid)

            label  = result["aria_label"] or "none"
            req_sat = result.get("requirement_satisfaction", 0.0)
            gc = result.get("gaia_correct")
            gc_str = ("✓" if gc else "✗") if gc is not None else "?"
            print(
                f"{tid:<38} {task['level']:>5} {label:<22} {req_sat:>6.2f}  {gc_str:>5}  OK"
            )

        _save_progress(progress)

        if i < len(tasks) - 1:
            time.sleep(delay)

    print("-" * 88)
    print(f"Batch {batch_num} done: {ran} ran | {skipped} skipped | {errors} errors")
    return {"ran": ran, "skipped": skipped, "errors": errors}


# ── CLI ───────────────────────────────────────────────────────────────────────

@click.command()
@click.option("--batch",      default=None, type=int,  help="Batch number to run (0-based)")
@click.option("--batch-size", default=5,   type=int,  show_default=True, help="Tasks per batch")
@click.option("--all",        "run_all",   is_flag=True, help="Run all batches automatically")
@click.option("--status",     is_flag=True, help="Show progress without running anything")
@click.option("--delay",      default=10,  type=float, show_default=True,
              help="Seconds between tasks within a batch (rate limit buffer)")
@click.option("--batch-delay", default=60, type=float, show_default=True,
              help="Seconds between batches when using --all (allows API key swap)")
@click.option("--level",      default=1,   type=int,  show_default=True,
              help="Filter by GAIA level (1, 2, or 0 for all)")
@click.option("--force",      is_flag=True, help="Re-run tasks already completed")
def main(batch, batch_size, run_all, status, delay, batch_delay, level, force):
    """Run GAIA benchmark tasks through the full ARIA pipeline."""
    tasks = _load_tasks()

    if level > 0:
        tasks = [t for t in tasks if t["level"] == level]

    total_batches = (len(tasks) + batch_size - 1) // batch_size

    if status:
        progress = _load_progress()
        print(f"\nGAIA Tasks: {len(tasks)} (Level {level if level else 'all'})")
        print(f"Completed:  {len(progress['completed'])}")
        print(f"Failed:     {len(progress['failed'])}")
        remaining = len(tasks) - len(set(progress["completed"]) & {t["gaia_task_id"] for t in tasks})
        print(f"Remaining:  {remaining}")
        print(f"Batches:    {total_batches} × {batch_size}")
        return

    print("\nARIA-GAIA Benchmark Runner")
    print("=" * 40)
    _preflight()

    if run_all:
        print(f"\nRunning all {total_batches} batches ({len(tasks)} tasks)")
        print(f"Batch delay: {batch_delay}s  (swap GROQ_API_KEY in .env between batches if needed)\n")
        for b in range(total_batches):
            batch_tasks = tasks[b * batch_size: (b + 1) * batch_size]
            _run_batch(batch_tasks, b, delay, force)
            if b < total_batches - 1:
                print(f"\n[Waiting {batch_delay}s before batch {b+1}...]")
                print("[You can now swap GROQ_API_KEY in .env if needed]")
                time.sleep(batch_delay)
        print(f"\nAll done. Next: python scripts/gaia_agreement.py")

    elif batch is not None:
        if batch >= total_batches:
            print(f"ERROR: batch {batch} out of range. Max batch: {total_batches - 1}")
            sys.exit(1)
        batch_tasks = tasks[batch * batch_size: (batch + 1) * batch_size]
        _run_batch(batch_tasks, batch, delay, force)
        next_batch = batch + 1
        if next_batch < total_batches:
            print(f"\nNext batch: python scripts/gaia_run_batch.py --batch {next_batch}")
            print("[Swap GROQ_API_KEY in .env now if needed]")
        else:
            print(f"\nAll batches complete. Next: python scripts/gaia_agreement.py")

    else:
        print("Specify --batch N, --all, or --status")
        print(f"Available batches: 0 to {total_batches - 1}")
        print(f"Example: python scripts/gaia_run_batch.py --batch 0")


if __name__ == "__main__":
    main()
