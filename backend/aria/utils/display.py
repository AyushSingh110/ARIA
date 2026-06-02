from __future__ import annotations

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box

console = Console()

_PHASE_COLORS = {
    "decompose": "cyan",
    "execute": "green",
    "observe": "yellow",
    "critique": "magenta",
    "diagnose": "red",
    "refine": "blue",
    "validate": "bright_cyan",
    "commit": "bright_green",
    "complete": "bright_green",
    "escalated": "bright_red",
}


def print_phase(phase: str, detail: str = "") -> None:
    color = _PHASE_COLORS.get(phase, "white")
    label = f"[bold {color}]▶ PHASE: {phase.upper()}[/bold {color}]"
    msg = f"{label}  {detail}" if detail else label
    console.print(msg)


def print_agent_output(agent_name: str, content: str, color: str = "white") -> None:
    console.print(
        Panel(
            content,
            title=f"[bold {color}]{agent_name}[/bold {color}]",
            border_style=color,
            expand=False,
        )
    )


def print_run_summary(state: dict) -> None:
    table = Table(title="ARIA Run Summary", box=box.ROUNDED, border_style="bright_blue")
    table.add_column("Field", style="bold cyan", no_wrap=True)
    table.add_column("Value", style="white")

    table.add_row("Task ID", state.get("task_id", "—")[:16] + "…")
    table.add_row("Task Class", state.get("task_class", "—"))
    table.add_row("Phase Reached", state.get("current_phase", "—"))
    table.add_row("Turns Used", str(state.get("executor_turn_count", 0)))
    table.add_row("Anomaly Detected", str(state.get("anomaly_detected", False)))
    table.add_row("Anomaly Severity", f"{state.get('anomaly_severity', 0.0):.3f}")
    table.add_row("Flags Raised", str(len(state.get("observer_flags", []))))
    table.add_row("Groq API Calls", str(state.get("api_calls_groq", 0)))
    table.add_row("Ollama API Calls", str(state.get("api_calls_ollama", 0)))
    table.add_row("Log Path", state.get("observer_log_path") or "—")

    console.print(table)


def print_observer_flags(flags: list[dict]) -> None:
    if not flags:
        console.print("[green]Observer: No anomalies detected.[/green]")
        return
    table = Table(title="Observer Flags", box=box.SIMPLE, border_style="yellow")
    table.add_column("Turn", style="dim")
    table.add_column("Type", style="bold yellow")
    table.add_column("Severity", style="red")
    table.add_column("Description")
    for f in flags:
        table.add_row(
            str(f.get("turn", "?")),
            f.get("flag_type", "?"),
            f"{f.get('signal_value', 0):.3f}",
            f.get("description", ""),
        )
    console.print(table)
