#!/usr/bin/env python3
"""Download and prepare the GAIA benchmark dataset.

Downloads from HuggingFace: gaia-benchmark/GAIA (validation split).
Filters to text-only tasks (no file attachments).
Saves to data/gaia/tasks.json.

Requirements:
  pip install datasets huggingface_hub

Auth:
  huggingface-cli login          (paste your HF token)
  -- OR --
  set HUGGINGFACE_TOKEN=hf_xxx in .env

Run:
  cd backend
  python scripts/gaia_download.py
  python scripts/gaia_download.py --all-levels   # include levels 2 and 3
  python scripts/gaia_download.py --with-files   # include tasks with file attachments
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import click

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

GAIA_DIR   = Path("data/gaia")
TASKS_FILE = GAIA_DIR / "tasks.json"


def _load_from_hf(all_levels: bool, with_files: bool) -> list[dict]:
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: 'datasets' not installed.  Run:  pip install datasets huggingface_hub")
        sys.exit(1)

    print("Downloading GAIA validation split from HuggingFace...")
    print("(If this fails with an auth error, run: huggingface-cli login)")

    try:
        ds = load_dataset(
            "gaia-benchmark/GAIA",
            "2023_all",
            split="validation",
            trust_remote_code=True,
        )
    except Exception as exc:
        print(f"\nERROR loading GAIA dataset: {exc}")
        print("\nMake sure you have:")
        print("  1. Accepted the dataset terms at https://huggingface.co/datasets/gaia-benchmark/GAIA")
        print("  2. Logged in with: huggingface-cli login")
        sys.exit(1)

    tasks = []
    for row in ds:
        level = int(row.get("Level", 1))
        file_name = (row.get("file_name") or "").strip()

        if not all_levels and level != 1:
            continue
        if not with_files and file_name:
            continue

        tasks.append({
            "gaia_task_id": row["task_id"],
            "question":     row["Question"],
            "level":        level,
            "expected_answer": str(row.get("Final answer", "")).strip(),
            "file_name":    file_name,
            "annotator_metadata": dict(row.get("Annotator Metadata") or {}),
            # ARIA fields
            "task_description": _format_task(row["Question"]),
            "task_class":       _infer_class(row["Question"], level),
        })

    return tasks


def _format_task(question: str) -> str:
    return (
        "Solve the following task step by step. Use available tools as needed.\n\n"
        f"Task: {question}"
    )


def _infer_class(question: str, level: int) -> str:
    q = question.lower()
    if any(w in q for w in ["search", "find", "look up", "who", "when", "where", "what year"]):
        return "web_research"
    if any(w in q for w in ["calculate", "compute", "how many", "sum", "percent", "average"]):
        return "reasoning"
    if any(w in q for w in ["write", "save", "create", "generate"]):
        return "code_or_write"
    return "general"


@click.command()
@click.option("--all-levels",  is_flag=True, help="Include Level 2 and 3 tasks (harder, slower)")
@click.option("--with-files",  is_flag=True, help="Include tasks that require file attachments")
def main(all_levels: bool, with_files: bool):
    """Download GAIA tasks and save to data/gaia/tasks.json."""
    GAIA_DIR.mkdir(parents=True, exist_ok=True)

    tasks = _load_from_hf(all_levels, with_files)

    by_level = {}
    for t in tasks:
        by_level.setdefault(t["level"], []).append(t)

    output = {"tasks": tasks, "meta": {
        "total": len(tasks),
        "by_level": {k: len(v) for k, v in sorted(by_level.items())},
        "with_files": with_files,
        "all_levels": all_levels,
    }}
    TASKS_FILE.write_text(json.dumps(output, indent=2), encoding="utf-8")

    print(f"\nSaved {len(tasks)} tasks to {TASKS_FILE}")
    for lvl, count in sorted(by_level.items()):
        print(f"  Level {lvl}: {count} tasks")
    print(f"\nNext: python scripts/gaia_run_batch.py --batch 0")


if __name__ == "__main__":
    main()
