#!/usr/bin/env python3
"""Human review interface for ARIA-RealBench results.

Shows each completed run and asks for a human label.
Saves the label back to the result file.

Run: python scripts/review_realbench.py
     python scripts/review_realbench.py --unreviewed-only
     python scripts/review_realbench.py --id rb_006
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import click

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

RESULTS_DIR = Path("data/realbench/results")

VALID_LABELS = [
    "none",
    "prompt_drift",
    "tool_misuse",
    "context_overflow",
    "goal_misalignment",
    "hallucination_loop",
    "gap",        # new failure type not in taxonomy
    "multi",      # multiple classes simultaneously
    "unclear",    # reviewer cannot determine
]

LABEL_MENU = "\n".join(
    f"  {i+1}. {lbl}" for i, lbl in enumerate(VALID_LABELS)
)


def load_result(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def save_result(path: Path, result: dict) -> None:
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")


def display_run(result: dict) -> None:
    sep  = "-" * 65
    sep2 = "=" * 65

    flags  = result.get("observer_flags", [])
    scores = result.get("critic_scores", {})
    flag_str = ", ".join(f["flag_type"] for f in flags) if flags else "none"

    print(f"\n{sep2}")
    print(f"  Task ID    : {result['task_id']}")
    print(f"  Task       : {result['task']}")
    print(f"  Task class : {result['task_class']}")
    print(f"  Expected   : {result.get('expected_class', '?')}")
    print(sep)
    print(f"  ARIA label      : {result.get('aria_label') or 'none'}")
    print(f"  ARIA confidence : {result.get('aria_confidence', 0):.2f}")
    print(f"  ARIA reasoning  : {result.get('aria_reasoning', '')[:200]}")
    print(sep)
    print(f"  Observer flags  : {flag_str}")
    print(f"  Turns used      : {result.get('executor_turn_count', 0)}")
    print(f"  Critic scores   : corr={scores.get('correctness','?')}  "
          f"comp={scores.get('completeness','?')}  "
          f"eff={scores.get('efficiency','?')}  "
          f"pass={scores.get('pass_fail','?')}")
    print(sep)
    print("  Trace:")
    for line in result.get("trace_summary", "").split("\n"):
        print(f"    {line[:100]}")
    print(sep)
    print("  Executor output:")
    print(f"    {result.get('executor_output', '')[:300]}")
    print(sep2)


def ask_label(current: str | None) -> tuple[str, str]:
    print(f"\n  Current human label: {current or 'not set'}")
    print(f"\n  Choose human label:")
    print(LABEL_MENU)
    print(f"  {len(VALID_LABELS)+1}. skip (keep current)")
    print(f"  {len(VALID_LABELS)+2}. quit review session")

    while True:
        raw = input("\n  Enter number or label name: ").strip().lower()

        if raw in ("q", "quit", str(len(VALID_LABELS)+2)):
            return "__quit__", ""

        if raw in ("s", "skip", str(len(VALID_LABELS)+1)):
            return "__skip__", ""

        # Number input
        try:
            idx = int(raw) - 1
            if 0 <= idx < len(VALID_LABELS):
                label = VALID_LABELS[idx]
                notes = input(f"  Notes (optional, press Enter to skip): ").strip()
                return label, notes
        except ValueError:
            pass

        # Direct label name
        if raw in VALID_LABELS:
            notes = input(f"  Notes (optional, press Enter to skip): ").strip()
            return raw, notes

        print(f"  Invalid input. Enter a number 1-{len(VALID_LABELS)+2} or a label name.")


@click.command()
@click.option("--unreviewed-only", is_flag=True, help="Only show unreviewed runs")
@click.option("--id", "task_id", default=None, help="Review single task by ID")
def main(unreviewed_only: bool, task_id: str | None):
    """Human review interface for ARIA-RealBench results."""
    if not RESULTS_DIR.exists():
        print("No results found. Run run_realbench.py first.")
        sys.exit(1)

    if task_id:
        files = [RESULTS_DIR / f"{task_id}.json"]
        files = [f for f in files if f.exists()]
        if not files:
            print(f"No result file for task '{task_id}'")
            sys.exit(1)
    else:
        files = sorted(RESULTS_DIR.glob("rb_*.json"))

    if unreviewed_only:
        files = [f for f in files if not load_result(f).get("reviewed")]

    if not files:
        print("No results to review.")
        sys.exit(0)

    reviewed = 0
    skipped  = 0

    print(f"\nReviewing {len(files)} runs. Type 'skip' to skip, 'quit' to stop.")

    for path in files:
        result = load_result(path)

        if result.get("run_error"):
            print(f"\n  Skipping {result['task_id']} (run error: {result['run_error'][:60]})")
            continue

        display_run(result)

        label, notes = ask_label(result.get("human_label"))

        if label == "__quit__":
            print(f"\nSession ended. Reviewed {reviewed} runs this session.")
            break
        elif label == "__skip__":
            skipped += 1
            continue
        else:
            result["human_label"] = label
            result["human_notes"] = notes if notes else None
            result["reviewed"]    = True
            save_result(path, result)
            reviewed += 1
            print(f"  Saved: {result['task_id']} -> {label}")

    total_reviewed = sum(1 for f in sorted(RESULTS_DIR.glob("rb_*.json"))
                         if load_result(f).get("reviewed"))
    total_files    = len(sorted(RESULTS_DIR.glob("rb_*.json")))
    print(f"\nTotal reviewed: {total_reviewed}/{total_files}")
    print("Run analyze_realbench.py when ready for distribution analysis.")


if __name__ == "__main__":
    main()
