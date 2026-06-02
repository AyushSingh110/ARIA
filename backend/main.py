#!/usr/bin/env python3
"""ARIA — Autonomous Reflective Intelligence Architecture
Phase 1 CLI entry point.
"""
from __future__ import annotations

import sys

import click
from rich.rule import Rule

from aria.config import get_settings
from aria.graph import build_graph
from aria.state import make_initial_state
from aria.utils.display import console, print_run_summary


@click.command()
@click.argument("task", type=str)
@click.option(
    "--task-class",
    default="general",
    show_default=True,
    help="Task class hint: code_generation | web_research | data_analysis | reasoning | tool_chaining | general",
)
@click.option(
    "--max-retries",
    default=None,
    type=int,
    help="Override max retries (default from .env)",
)
@click.option("--verbose", is_flag=True, help="Show full final state dump.")
def run(task: str, task_class: str, max_retries: int | None, verbose: bool) -> None:
    """Run an ARIA task through the Orchestrator → Executor → Observer pipeline.

    TASK is the natural-language task description (wrap in quotes if it contains spaces).

    Examples:

    \b
      python main.py "Calculate the compound interest on $10,000 at 5% for 3 years"
      python main.py "Search for information about LangGraph and write a summary to summary.txt"
      python main.py "What is 2 to the power of 32?" --task-class reasoning
    """
    settings = get_settings()

    if settings.orchestrator_provider == "groq" and not settings.groq_api_key:
        console.print(
            "[bold red]Error:[/bold red] GROQ_API_KEY is not set. "
            "Copy .env.example → .env and add your key, or set ORCHESTRATOR_PROVIDER=ollama."
        )
        sys.exit(1)

    console.print(Rule("[bold blue]ARIA — Phase 1[/bold blue]"))
    console.print(f"[bold]Task:[/bold] {task}")
    console.print(
        f"[dim]Orchestrator: {settings.orchestrator_provider} | "
        f"Executor: {settings.executor_provider}[/dim]\n"
    )

    retries = max_retries if max_retries is not None else settings.max_retries
    initial_state = make_initial_state(
        task_description=task,
        task_class=task_class,
        max_retries=retries,
    )

    graph = build_graph()

    try:
        final_state = graph.invoke(initial_state)
    except KeyboardInterrupt:
        console.print("\n[yellow]Run interrupted by user.[/yellow]")
        sys.exit(0)
    except Exception as exc:
        console.print(f"\n[bold red]Fatal error:[/bold red] {exc}")
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

    console.print(Rule("[bold blue]Run Complete[/bold blue]"))
    print_run_summary(final_state)

    if verbose:
        import json
        safe_state = {
            k: v for k, v in final_state.items()
            if k not in ("task_description_embedding", "goal_embedding_history", "messages")
        }
        console.print_json(json.dumps(safe_state, indent=2, default=str))


if __name__ == "__main__":
    run()
