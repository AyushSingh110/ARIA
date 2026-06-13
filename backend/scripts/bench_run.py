#!/usr/bin/env python3
"""Run any common-schema benchmark through the full ARIA pipeline.

This is the Step 1.2 trace-generation engine. It varies three knobs the
research design cares about and tags every trace with the config used, so
the weak-vs-strong contrast (Section 3) is recoverable later:

  --executor-provider {groq,ollama}   which backend runs the Executor
  --executor-model    <name>          e.g. weak  llama-3.1-8b-instant
                                            strong llama-3.3-70b-versatile
  --max-turns         <int>           turn budget (3 / 8 / 15)

Resumability (Groq rate-limit safe):
  * Each result is written to its own file immediately after the task runs.
  * The filename encodes the run config, so weak/strong and different turn
    budgets never overwrite each other.
  * On restart the runner skips any (task, config) whose result file already
    exists. Abort with Ctrl-C any time and re-run the same command — it
    continues from where it paused.
  * Transient Groq 429s are retried with exponential backoff inside the run;
    a task that still fails is recorded with run_error and picked up next time.

Examples:
  # weak executor, budget 8, first 60 GSM8K tasks
  python scripts/bench_run.py --benchmark gsm8k \
      --executor-provider groq --executor-model llama-3.1-8b-instant \
      --max-turns 8 --limit 60

  # strong executor, budget 15
  python scripts/bench_run.py --benchmark hotpotqa \
      --executor-provider groq --executor-model llama-3.3-70b-versatile \
      --max-turns 15

  # progress for a given config
  python scripts/bench_run.py --benchmark gsm8k \
      --executor-model llama-3.1-8b-instant --max-turns 8 --status
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import click

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

DATA_ROOT = Path("data")


# ── Answer matching (shared logic with gaia_run_batch) ──────────────────────

def _answer_matches(expected: str, output: str) -> bool | None:
    if not expected or not output:
        return None
    exp = expected.lower().strip()
    out = output.lower()
    if exp in out:
        return True
    _ABBREV = {"saint": "st", "street": "st", "mount": "mt", "doctor": "dr", "fort": "ft"}
    for full, short in _ABBREV.items():
        if exp.replace(full, short) in out or exp.replace(short, full) in out:
            return True
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
    if re.search(r"\b" + re.escape(exp) + r"\b", out):
        return True
    return False


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


# ── Run config ──────────────────────────────────────────────────────────────

def _model_slug(provider: str, model: str) -> str:
    base = model or ("default-" + provider)
    return re.sub(r"[^a-zA-Z0-9]+", "-", base).strip("-").lower()


def _config_suffix(provider: str, model: str, max_turns: int) -> str:
    return f"{provider}-{_model_slug(provider, model)}__t{max_turns}"


def _apply_run_config(provider: str, model: str, max_turns: int) -> None:
    """Push the run config into the environment and rebuild settings."""
    from aria.config import get_settings
    os.environ["EXECUTOR_PROVIDER"] = provider
    os.environ["EXECUTOR_MODEL"] = model or ""
    os.environ["EXECUTOR_MAX_TURNS"] = str(max_turns)
    get_settings.cache_clear()        # force re-read of env on next call


# ── Preflight ────────────────────────────────────────────────────────────────

def _preflight(provider: str) -> None:
    from aria.config import get_settings
    import httpx
    s = get_settings()
    if not s.groq_api_key:
        print("ERROR: GROQ_API_KEY not set in .env (orchestrator/critic need it).")
        sys.exit(1)
    needs_ollama = "ollama" in (provider, s.orchestrator_provider, s.critic_provider)
    if needs_ollama:
        try:
            httpx.get(f"{s.ollama_base_url}/api/tags", timeout=3.0)
        except Exception:
            print(f"ERROR: Ollama selected but not reachable at {s.ollama_base_url}.")
            sys.exit(1)


# ── Single task ───────────────────────────────────────────────────────────────

def _run_one(task: dict, cfg: dict, max_429_retries: int = 4) -> dict:
    from aria.graph import build_graph
    from aria.state import make_initial_state

    attempt, run_error, final_state = 0, None, None
    while True:
        initial_state = make_initial_state(
            task_description=task["task_description"],
            task_class=task.get("task_class", "general"),
            max_retries=1,
        )
        try:
            final_state = build_graph().invoke(initial_state)
            run_error = None
            break
        except Exception as exc:
            msg = str(exc)
            is_rate_limit = "429" in msg or "rate limit" in msg.lower() or "rate_limit" in msg.lower()
            if is_rate_limit and attempt < max_429_retries:
                backoff = min(60, 5 * (2 ** attempt))
                print(f"      [429] rate limited — backoff {backoff}s "
                      f"(attempt {attempt + 1}/{max_429_retries})")
                time.sleep(backoff)
                attempt += 1
                continue
            final_state = initial_state
            run_error = msg
            break

    flags  = [dict(f) for f in (final_state.get("observer_flags") or [])]
    scores = dict(final_state.get("critic_scores") or {})
    trace  = [dict(e) for e in (final_state.get("executor_trace") or [])]
    output = final_state.get("executor_output") or ""

    return {
        # identity
        "task_id":        task["id"],
        "benchmark":      task.get("benchmark"),
        "task":           task.get("task"),
        "task_class":     task.get("task_class"),
        "level":          task.get("level"),
        "expected_answer": task.get("expected_answer", ""),

        # run config (the Step 1.2 contrast knobs)
        "run_config": cfg,

        # ARIA diagnosis
        "aria_label":          final_state.get("failure_class"),
        "aria_confidence":     final_state.get("diagnosis_confidence"),
        "aria_reasoning":      (final_state.get("diagnosis_reasoning") or "")[:400],
        "aria_manifestation":  final_state.get("failure_manifestation"),
        "requirement_satisfaction": final_state.get("requirement_satisfaction", 0.0),

        # observer / executor / critic
        "observer_flags":      flags,
        "anomaly_detected":    final_state.get("anomaly_detected", False),
        "anomaly_severity":    final_state.get("anomaly_severity", 0.0),
        "drift_scores":        final_state.get("drift_scores") or [],
        "executor_output":     output[:600],
        "executor_turn_count": final_state.get("executor_turn_count", 0),
        "trace":               trace,
        "trace_summary":       _summarise(trace),
        "critic_scores":       scores,
        "grounding": dict(final_state["grounding"]) if final_state.get("grounding") else None,

        # answer check
        "answer_correct": _answer_matches(task.get("expected_answer", ""), output),

        # human review (filled later) — keep test set real & untouched
        "human_label": None, "human_notes": None, "reviewed": False,

        "run_error":     run_error,
        "run_timestamp": datetime.now(timezone.utc).isoformat(),
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

@click.command()
@click.option("--benchmark", required=True, help="Benchmark name (matches data/<bench>/tasks.json)")
@click.option("--executor-provider", default="groq",
              type=click.Choice(["groq", "ollama"]), show_default=True)
@click.option("--executor-model", default="llama-3.1-8b-instant", show_default=True,
              help="Executor model. Weak: llama-3.1-8b-instant | Strong: llama-3.3-70b-versatile")
@click.option("--max-turns", default=8, type=int, show_default=True, help="Executor turn budget")
@click.option("--limit", default=0, type=int, help="Max tasks from the file (0 = all)")
@click.option("--start", default=0, type=int, help="Start index into the task list")
@click.option("--delay", default=8.0, type=float, show_default=True,
              help="Seconds between tasks (Groq rate-limit buffer)")
@click.option("--status", is_flag=True, help="Show progress for this config and exit")
@click.option("--force", is_flag=True, help="Re-run tasks even if a result file exists")
def main(benchmark, executor_provider, executor_model, max_turns,
         limit, start, delay, status, force):
    """Generate ARIA traces for a benchmark under a specific executor config."""
    tasks_path = DATA_ROOT / benchmark / "tasks.json"
    if not tasks_path.exists():
        print(f"ERROR: {tasks_path} not found. Run: python scripts/bench_download.py {benchmark}")
        sys.exit(1)

    tasks = json.loads(tasks_path.read_text(encoding="utf-8"))["tasks"]
    tasks = tasks[start:]
    if limit:
        tasks = tasks[:limit]

    suffix = _config_suffix(executor_provider, executor_model, max_turns)
    results_dir = DATA_ROOT / benchmark / "results"
    cfg = {
        "executor_provider": executor_provider,
        "executor_model": executor_model or f"default-{executor_provider}",
        "max_turns": max_turns,
        "config_id": suffix,
    }

    def _done(tid: str) -> bool:
        # Succeeded results count as done; errored files are re-tried on the
        # next run so an aborted / rate-limited pass resumes without --force.
        fp = results_dir / f"{tid}__{suffix}.json"
        if not fp.exists():
            return False
        try:
            return not json.loads(fp.read_text(encoding="utf-8")).get("run_error")
        except Exception:
            return False

    if status:
        done = sum(1 for t in tasks if _done(t["id"]))
        print(f"\n{benchmark} | config={suffix}")
        print(f"  Tasks:     {len(tasks)}")
        print(f"  Completed: {done}")
        print(f"  Remaining: {len(tasks) - done}")
        return

    print("\nARIA Trace Generator")
    print("=" * 60)
    _apply_run_config(executor_provider, executor_model, max_turns)
    _preflight(executor_provider)
    print(f"  Benchmark : {benchmark}  ({len(tasks)} tasks)")
    print(f"  Executor  : {executor_provider} / {executor_model or 'default'}  | max_turns={max_turns}")
    print(f"  Config id : {suffix}")
    print(f"  Results   : {results_dir}/")
    results_dir.mkdir(parents=True, exist_ok=True)

    ran = skipped = errors = 0
    print(f"\n{'task':<16}{'label':<20}{'turns':>5}{'ans':>5}  status")
    print("-" * 60)

    for i, task in enumerate(tasks):
        tid = task["id"]
        if not force and _done(tid):
            skipped += 1
            continue
        try:
            result = _run_one(task, cfg)
        except KeyboardInterrupt:
            print("\n[aborted — progress saved; re-run the same command to resume]")
            break

        (results_dir / f"{tid}__{suffix}.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

        if result["run_error"]:
            errors += 1
            print(f"{tid:<16}{'ERROR':<20}{'':>5}{'':>5}  {result['run_error'][:30]}")
        else:
            ran += 1
            label = result["aria_label"] or "none"
            ac = result["answer_correct"]
            ac_s = ("ok" if ac else "x") if ac is not None else "?"
            print(f"{tid:<16}{label:<20}{result['executor_turn_count']:>5}{ac_s:>5}  OK")

        if i < len(tasks) - 1 and not result["run_error"]:
            time.sleep(delay)

    print("-" * 60)
    print(f"Done: {ran} ran | {skipped} skipped | {errors} errors")
    print(f"Resume/extend: re-run the same command (skips completed).")


if __name__ == "__main__":
    main()
