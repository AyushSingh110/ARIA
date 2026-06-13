#!/usr/bin/env python3
"""Run tau-bench episodes through ARIA (rollout + diagnosis), resumable.

For each task this:
  1. rolls out a tau-bench episode (aria.adapters.taubench_adapter.run_episode)
     using a configurable agent model (weak/strong) and turn budget,
  2. feeds the resulting trace through ARIA's Observer -> Critic -> Diagnostician
     (the same diagnosis every other benchmark gets),
  3. records the env's ground-truth reward as the success signal,
  4. saves one result file per (task, config) so weak/strong/budget variants
     never collide and the run resumes after a Ctrl-C / Groq-429 abort.

NOTE: tau-bench makes Groq calls for BOTH the agent and the simulated user, so
this is the most rate-limit-intensive runner — keep --delay generous and rely
on resume. Requires GROQ_API_KEY and the vendored third_party/tau-bench.

Examples:
  python scripts/taubench_run.py --env retail --agent-model llama-3.1-8b-instant \
      --max-turns 15 --limit 20
  python scripts/taubench_run.py --env airline --max-turns 15 --status
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


def _slug(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "-", s).strip("-").lower()


def _summarise(trace: list[dict]) -> str:
    lines = []
    for e in trace:
        if e.get("tool_name") in ("__llm__", "respond"):
            lines.append(f"turn {e['turn']}: [{e.get('tool_name')}] {str(e.get('llm_output') or e.get('tool_result',''))[:80]}")
        else:
            lines.append(
                f"turn {e['turn']}: {e.get('tool_name')}("
                f"{json.dumps(e.get('tool_args', {}))[:40]}) -> {str(e.get('tool_result', ''))[:60]}")
    return "\n".join(lines)


def _diagnose(task_instruction: str, episode: dict) -> dict:
    """Run the trace through ARIA Observer -> Critic -> Diagnostician."""
    from aria.agents.observer import observer_node
    from aria.agents.critic import critic_node
    from aria.agents.diagnostician import diagnostician_node
    from aria.config import get_settings
    from aria.memory.embeddings import get_engine
    from aria.state import make_initial_state

    settings = get_settings()
    engine = get_engine(settings.embedding_model)

    state = make_initial_state(task_description=task_instruction, task_class="tool_interaction")
    state["task_description_embedding"] = engine.embed(task_instruction[:512] or "task")
    state["executor_trace"] = episode["trace"]
    state["executor_output"] = episode["executor_output"]
    state["executor_turn_count"] = episode["turn_count"]
    state["goal_embedding_history"] = episode["goal_embedding_history"]
    state["active_subtask"] = {"id": "s1", "description": task_instruction,
                               "expected_tools": [], "success_criteria": "resolve user request"}
    state["subtasks"] = [state["active_subtask"]]

    for node in (observer_node, critic_node, diagnostician_node):
        try:
            state.update(node(state))
        except Exception as exc:
            print(f"      [diagnosis node {node.__name__} failed: {str(exc)[:80]}]")
    return state


@click.command()
@click.option("--env", "env_name", default="retail",
              type=click.Choice(["retail", "airline"]), show_default=True)
@click.option("--agent-provider", default="groq", type=click.Choice(["groq", "ollama"]), show_default=True)
@click.option("--agent-model", default="llama-3.1-8b-instant", show_default=True,
              help="Weak: llama-3.1-8b-instant | Strong: llama-3.3-70b-versatile")
@click.option("--user-model", default="llama-3.3-70b-versatile", show_default=True,
              help="Model that simulates the user (Groq via litellm)")
@click.option("--max-turns", default=15, type=int, show_default=True)
@click.option("--task-split", default="test", show_default=True)
@click.option("--start", default=0, type=int, help="First task index")
@click.option("--limit", default=20, type=int, help="Number of task indices to run")
@click.option("--delay", default=10.0, type=float, show_default=True,
              help="Seconds between episodes (Groq rate-limit buffer)")
@click.option("--status", is_flag=True, help="Show progress for this config and exit")
@click.option("--force", is_flag=True, help="Re-run even if a result file exists")
def main(env_name, agent_provider, agent_model, user_model, max_turns, task_split,
         start, limit, delay, status, force):
    """Generate + diagnose tau-bench traces under one agent config."""
    from aria.adapters.taubench_adapter import run_episode
    from aria.config import get_settings

    suffix = f"{env_name}__{agent_provider}-{_slug(agent_model)}__t{max_turns}"
    results_dir = DATA_ROOT / "taubench" / "results"
    cfg = {
        "benchmark": "taubench", "env": env_name,
        "agent_provider": agent_provider, "agent_model": agent_model,
        "user_model": user_model, "max_turns": max_turns, "config_id": suffix,
    }
    indices = list(range(start, start + limit))

    def _done(idx: int) -> bool:
        # A result counts as done only if it succeeded; errored files are
        # re-tried on the next run so an aborted/rate-limited pass resumes
        # cleanly without --force.
        fp = results_dir / f"{env_name}_{idx:04d}__{suffix}.json"
        if not fp.exists():
            return False
        try:
            return not json.loads(fp.read_text(encoding="utf-8")).get("run_error")
        except Exception:
            return False

    if status:
        done = sum(1 for i in indices if _done(i))
        print(f"\ntaubench/{env_name} | config={suffix}")
        print(f"  Tasks: {len(indices)}  Completed: {done}  Remaining: {len(indices) - done}")
        return

    if not get_settings().groq_api_key:
        print("ERROR: GROQ_API_KEY not set.")
        sys.exit(1)

    print("\nARIA × tau-bench Trace Generator")
    print("=" * 60)
    print(f"  Env       : {env_name}  ({len(indices)} tasks)")
    print(f"  Agent     : {agent_provider}/{agent_model}  | user-sim: {user_model} | max_turns={max_turns}")
    print(f"  Config id : {suffix}")
    results_dir.mkdir(parents=True, exist_ok=True)

    ran = skipped = errors = 0
    print(f"\n{'task':<14}{'aria_label':<20}{'turns':>5}{'reward':>7}  status")
    print("-" * 60)

    for n, idx in enumerate(indices):
        if not force and _done(idx):
            skipped += 1
            continue
        try:
            episode = run_episode(
                env_name, idx, agent_provider=agent_provider, agent_model=agent_model,
                user_model=user_model, task_split=task_split, max_turns=max_turns)
            run_error = episode.get("run_error")
        except KeyboardInterrupt:
            print("\n[aborted — progress saved; re-run the same command to resume]")
            break
        except Exception as exc:
            msg = str(exc)
            # Out-of-range task index → we've run the whole split.
            if "index" in msg.lower() and ("range" in msg.lower() or "out of" in msg.lower()):
                print(f"{env_name}_{idx:04d}   (no more tasks in split)")
                break
            episode, run_error = None, msg

        if episode is None:
            errors += 1
            (results_dir / f"{env_name}_{idx:04d}__{suffix}.json").write_text(
                json.dumps({"task_id": f"{env_name}_{idx:04d}", "run_config": cfg,
                            "run_error": run_error,
                            "run_timestamp": datetime.now(timezone.utc).isoformat()},
                           indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"{env_name}_{idx:04d}   {'ERROR':<20}{'':>5}{'':>7}  {run_error[:24]}")
            if n < len(indices) - 1:
                time.sleep(delay)
            continue

        diag = _diagnose(episode["task_instruction"], episode)
        reward = episode["reward"]
        result = {
            "task_id": f"{env_name}_{idx:04d}",
            "benchmark": "taubench",
            "env": env_name,
            "task_index": idx,
            "task": episode["task_instruction"],
            "task_class": "tool_interaction",
            "run_config": cfg,
            # success signal = env ground-truth reward
            "reward": reward,
            "answer_correct": bool(reward >= 1.0),
            "episode_done": episode["done"],
            # ARIA diagnosis
            "aria_label": diag.get("failure_class"),
            "aria_confidence": diag.get("diagnosis_confidence"),
            "aria_reasoning": (diag.get("diagnosis_reasoning") or "")[:400],
            "aria_manifestation": diag.get("failure_manifestation"),
            "observer_flags": [dict(f) for f in (diag.get("observer_flags") or [])],
            "anomaly_detected": diag.get("anomaly_detected", False),
            "drift_scores": diag.get("drift_scores") or [],
            "critic_scores": dict(diag.get("critic_scores") or {}),
            "requirement_satisfaction": diag.get("requirement_satisfaction", 0.0),
            # executor view
            "executor_output": (episode["executor_output"] or "")[:600],
            "executor_turn_count": episode["turn_count"],
            "trace": episode["trace"],
            "trace_summary": _summarise(episode["trace"]),
            # human review (kept real)
            "human_label": None, "human_notes": None, "reviewed": False,
            "run_error": None,
            "run_timestamp": datetime.now(timezone.utc).isoformat(),
        }
        (results_dir / f"{env_name}_{idx:04d}__{suffix}.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        ran += 1
        label = result["aria_label"] or "none"
        print(f"{env_name}_{idx:04d}   {label:<20}{episode['turn_count']:>5}{reward:>7.1f}  OK")

        if n < len(indices) - 1:
            time.sleep(delay)

    print("-" * 60)
    print(f"Done: {ran} ran | {skipped} skipped | {errors} errors")
    print("Resume/extend: re-run the same command (skips completed).")


if __name__ == "__main__":
    main()
