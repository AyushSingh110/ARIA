#!/usr/bin/env python3
"""ARIA — Autonomous Reflective Intelligence Architecture."""
from __future__ import annotations

import sys
import warnings

# Suppress noisy internal import warnings from LangGraph / LiteLLM
warnings.filterwarnings("ignore", message=".*JsonPlusSerializer.*")
warnings.filterwarnings("ignore", message=".*pkg_resources.*", category=DeprecationWarning)

import click
from rich.rule import Rule

from aria.config import get_settings
from aria.graph import build_graph
from aria.state import make_initial_state
from aria.utils.display import console, print_run_summary


def _preflight_check(settings) -> None:
    if settings.orchestrator_provider == "groq" and not settings.groq_api_key:
        console.print(
            "[bold red]Error:[/bold red] GROQ_API_KEY not set. "
            "Add it to .env or set ORCHESTRATOR_PROVIDER=ollama."
        )
        sys.exit(1)

    # Check Ollama is reachable if any provider uses it
    ollama_providers = [
        p for p in [settings.orchestrator_provider, settings.executor_provider, settings.critic_provider]
        if p == "ollama"
    ]
    if ollama_providers:
        import httpx
        try:
            httpx.get(f"{settings.ollama_base_url}/api/tags", timeout=3.0)
        except Exception:
            console.print(
                f"[bold red]Error:[/bold red] Cannot reach Ollama at [cyan]{settings.ollama_base_url}[/cyan].\n"
                "[yellow]Fix options:[/yellow]\n"
                "  1. Start Ollama in a separate terminal:  [bold]ollama serve[/bold]\n"
                "  2. Switch all providers to Groq in .env:\n"
                "     [dim]EXECUTOR_PROVIDER=groq\n"
                "     CRITIC_PROVIDER=groq[/dim]"
            )
            sys.exit(1)


@click.command()
@click.argument("task", type=str)
@click.option(
    "--task-class",
    default="general",
    show_default=True,
    help="code_generation | web_research | data_analysis | reasoning | tool_chaining | general",
)
@click.option("--max-retries", default=None, type=int, help="Override MAX_RETRIES from .env")
@click.option("--verbose", is_flag=True, help="Print full final state.")
def run(task: str, task_class: str, max_retries: int | None, verbose: bool) -> None:
    """Run a task through the full ARIA pipeline.

    \b
    Examples:
      python main.py "Calculate compound interest on $10,000 at 5% for 3 years"
      python main.py "Search for LangGraph info and write a summary to summary.txt"
      python main.py "What is 2 to the power of 32?" --task-class reasoning
    """
    settings = get_settings()
    _preflight_check(settings)

    console.print(Rule("[bold blue]ARIA — Phase 3[/bold blue]"))
    console.print(f"[bold]Task:[/bold] {task}")
    console.print(
        f"[dim]Orchestrator: {settings.orchestrator_provider} | "
        f"Executor: {settings.executor_provider} | "
        f"Critic: {settings.critic_provider}[/dim]\n"
    )

    initial_state = make_initial_state(
        task_description=task,
        task_class=task_class,
        max_retries=max_retries if max_retries is not None else settings.max_retries,
    )

    graph = build_graph()

    try:
        final_state = graph.invoke(initial_state)
    except KeyboardInterrupt:
        console.print("\n[yellow]Run interrupted.[/yellow]")
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
