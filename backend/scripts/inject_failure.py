#!/usr/bin/env python3
"""Phase 1 validation script — inject known failures and verify Observer captures them.

Three scenarios:
  1. tool_repetition  — Executor calls the same tool twice with identical args
  2. turn_budget      — Executor uses ≥ 80 % of its turn budget
  3. prompt_drift     — Executor output embeddings diverge from the goal

Run: python scripts/inject_failure.py [--scenario all|tool_repetition|turn_budget|prompt_drift]
"""
from __future__ import annotations

import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import click
from rich.rule import Rule

# Allow running from project root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aria.agents.observer import observer_node
from aria.config import get_settings
from aria.memory.embeddings import get_engine
from aria.state.schema import ARIAState, ExecutorTraceEntry, make_initial_state
from aria.utils.display import console, print_observer_flags, print_run_summary


# ── Helper: build a minimal ARIAState for injection ──────────────────────────

def _base_state(task: str) -> ARIAState:
    settings = get_settings()
    engine = get_engine(settings.embedding_model)
    state = make_initial_state(task_description=task, task_class="general")
    state["task_description_embedding"] = engine.embed(task)
    state["subtasks"] = [{"id": "s1", "description": task, "expected_tools": [], "success_criteria": "done"}]
    state["active_subtask"] = state["subtasks"][0]
    state["current_subtask_index"] = 0
    state["task_class"] = "general"
    return state


# ── Scenario 1: Tool repetition ───────────────────────────────────────────────

def scenario_tool_repetition() -> bool:
    console.print(Rule("[yellow]Scenario: tool_repetition[/yellow]"))
    task = "Calculate the area of a circle with radius 5"
    state = _base_state(task)
    settings = get_settings()
    engine = get_engine(settings.embedding_model)

    trace: list[ExecutorTraceEntry] = [
        ExecutorTraceEntry(turn=0, tool_name="calculator", tool_args={"expression": "3.14159 * 5 ** 2"},
                           tool_result="78.53975", llm_output="Let me calculate…", latency_ms=120, token_count=40),
        ExecutorTraceEntry(turn=1, tool_name="calculator", tool_args={"expression": "3.14159 * 5 ** 2"},
                           tool_result="78.53975", llm_output="Let me recalculate to be sure…", latency_ms=115, token_count=38),
        ExecutorTraceEntry(turn=2, tool_name="__llm__", tool_args={}, tool_result="",
                           llm_output="FINAL ANSWER: The area is 78.54 square units.", latency_ms=200, token_count=30),
    ]
    embs = [engine.embed(e["llm_output"]) for e in trace]

    state["executor_trace"] = trace
    state["executor_turn_count"] = 3
    state["executor_output"] = "FINAL ANSWER: The area is 78.54 square units."
    state["goal_embedding_history"] = embs
    state["current_phase"] = "observe"

    result = observer_node(state)
    flags = result.get("observer_flags", [])
    passed = any(f["flag_type"] == "tool_repetition" for f in flags)
    console.print(f"[{'green' if passed else 'red'}]tool_repetition detected: {passed}[/]")
    return passed


# ── Scenario 2: Turn budget warning ───────────────────────────────────────────

def scenario_turn_budget() -> bool:
    console.print(Rule("[yellow]Scenario: turn_budget[/yellow]"))
    settings = get_settings()
    engine = get_engine(settings.embedding_model)
    task = "Search the web for 10 different facts about Python and write each one to a separate file"
    state = _base_state(task)

    max_turns = settings.executor_max_turns
    used_turns = int(max_turns * 0.9)  # 90% — above 80% threshold

    trace: list[ExecutorTraceEntry] = [
        ExecutorTraceEntry(turn=i, tool_name="web_search", tool_args={"query": f"Python fact {i}"},
                           tool_result=f"Fact {i}: Python was created by Guido van Rossum.",
                           llm_output=f"Searching for fact {i}…", latency_ms=200, token_count=50)
        for i in range(used_turns)
    ]
    embs = [engine.embed(e["llm_output"]) for e in trace]

    state["executor_trace"] = trace
    state["executor_turn_count"] = used_turns
    state["executor_output"] = "Did not finish."
    state["goal_embedding_history"] = embs
    state["current_phase"] = "observe"

    result = observer_node(state)
    flags = result.get("observer_flags", [])
    passed = any(f["flag_type"] == "turn_budget_warning" for f in flags)
    console.print(f"[{'green' if passed else 'red'}]turn_budget_warning detected: {passed}[/]")
    return passed


# ── Scenario 3: Prompt drift ──────────────────────────────────────────────────

def scenario_prompt_drift() -> bool:
    console.print(Rule("[yellow]Scenario: prompt_drift[/yellow]"))
    settings = get_settings()
    engine = get_engine(settings.embedding_model)
    task = "Write a Python function that parses CSV files and returns a list of dicts"
    state = _base_state(task)

    # Simulate an agent that starts on-topic then drifts to argparse CLI setup
    turn_texts = [
        "I will write a CSV parser function using Python's csv module.",
        "Now adding type validation for robustness.",
        "Adding argparse so users can run this from the command line.",
        "Let me add logging setup and a README as well.",
        "FINAL ANSWER: Here is a full CLI application with argparse.",
    ]
    embs = [engine.embed(t) for t in turn_texts]
    trace: list[ExecutorTraceEntry] = [
        ExecutorTraceEntry(turn=i, tool_name="__llm__", tool_args={}, tool_result="",
                           llm_output=t, latency_ms=300, token_count=60)
        for i, t in enumerate(turn_texts)
    ]

    state["executor_trace"] = trace
    state["executor_turn_count"] = len(trace)
    state["executor_output"] = turn_texts[-1]
    state["goal_embedding_history"] = embs
    state["current_phase"] = "observe"

    result = observer_node(state)
    flags = result.get("observer_flags", [])
    passed = any(f["flag_type"] == "prompt_drift" for f in flags)
    console.print(f"[{'green' if passed else 'red'}]prompt_drift detected: {passed}[/]")
    return passed


# ── CLI ───────────────────────────────────────────────────────────────────────

_SCENARIOS = {
    "tool_repetition": scenario_tool_repetition,
    "turn_budget": scenario_turn_budget,
    "prompt_drift": scenario_prompt_drift,
}


@click.command()
@click.option(
    "--scenario",
    default="all",
    type=click.Choice(["all"] + list(_SCENARIOS.keys())),
    show_default=True,
    help="Which failure scenario to inject.",
)
def main(scenario: str) -> None:
    """Inject synthetic failures and verify Observer captures the correct signals."""
    console.print(Rule("[bold blue]ARIA Phase 1 — Failure Injection Validation[/bold blue]"))

    to_run = list(_SCENARIOS.items()) if scenario == "all" else [(scenario, _SCENARIOS[scenario])]
    results: dict[str, bool] = {}

    for name, fn in to_run:
        try:
            results[name] = fn()
        except Exception as exc:
            console.print(f"[red]Scenario '{name}' raised an exception: {exc}[/red]")
            results[name] = False
        console.print()

    console.print(Rule("Results"))
    all_passed = True
    for name, passed in results.items():
        icon = "✓" if passed else "✗"
        color = "green" if passed else "red"
        console.print(f"  [{color}]{icon} {name}[/{color}]")
        if not passed:
            all_passed = False

    if all_passed:
        console.print("\n[bold green]All scenarios passed — Phase 1 Observer is capturing correctly.[/bold green]")
    else:
        console.print("\n[bold red]Some scenarios failed — check Observer logic.[/bold red]")
        sys.exit(1)


if __name__ == "__main__":
    main()
