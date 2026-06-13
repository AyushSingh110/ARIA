#!/usr/bin/env python3
"""Download SWE-bench Lite into ARIA's task schema (trace-generation view).

SWE-bench Lite = 300 real GitHub bug-fix tasks. It is the strongest natural
source of the two rare classes (prompt_drift, context_overflow) because the
tasks are genuinely long-horizon. Unlike GSM8K/HotpotQA it cannot run on the
4 generic tools — the SWE-bench adapter (scripts/swebench_run.py) checks out
the repo and exposes repo-aware tools.

This downloader only materialises the task metadata. The heavy part (cloning
repos, running tests) happens in the runner / official harness.

Schema (data/swebench/tasks.json):
  {
    "id": "swebench_<instance_id>",
    "benchmark": "swebench_lite",
    "instance_id": "astropy__astropy-12907",
    "repo": "astropy/astropy",
    "base_commit": "...",
    "environment_setup_commit": "...",
    "problem_statement": "...",
    "test_patch": "...",            # gold tests (used ONLY for scoring)
    "fail_to_pass": [...],          # tests that should flip to passing
    "pass_to_pass": [...],          # tests that must stay passing
    "task_class": "code_generation",
    "task_description": "<prompt the executor sees>",
    "expected_answer": ""           # success = tests resolve, not string match
  }

Run:
  python scripts/swebench_download.py --limit 50
  python scripts/swebench_download.py            # all 300
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import click

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

OUT = Path("data/swebench/tasks.json")
HF_ID = "princeton-nlp/SWE-bench_Lite"


def _as_list(x) -> list:
    if isinstance(x, str):
        try:
            return json.loads(x)
        except Exception:
            return [x]
    return list(x) if x else []


def _prompt(repo: str, problem: str) -> str:
    return (
        "You are fixing a real bug in the checked-out repository "
        f"`{repo}`. Investigate the codebase with the available tools "
        "(list_repo, read_repo_file, search_repo, edit_repo_file, run_tests), "
        "make the minimal change that resolves the issue, and verify with "
        "run_tests. When the fix is complete, state \"FINAL ANSWER: done\".\n\n"
        f"Issue:\n{problem}"
    )


@click.command()
@click.option("--limit", default=0, type=int, help="Max instances (0 = all 300)")
def main(limit: int) -> None:
    """Materialise SWE-bench Lite task metadata."""
    from datasets import load_dataset
    print(f"Loading {HF_ID} (test split)...")
    ds = load_dataset(HF_ID, split="test")
    tasks = []
    for i, row in enumerate(ds):
        if limit and i >= limit:
            break
        tasks.append({
            "id": f"swebench_{row['instance_id']}",
            "benchmark": "swebench_lite",
            "instance_id": row["instance_id"],
            "repo": row["repo"],
            "base_commit": row["base_commit"],
            "environment_setup_commit": row.get("environment_setup_commit") or row["base_commit"],
            "problem_statement": row["problem_statement"],
            "patch": row.get("patch", ""),
            "test_patch": row.get("test_patch", ""),
            "fail_to_pass": _as_list(row.get("FAIL_TO_PASS")),
            "pass_to_pass": _as_list(row.get("PASS_TO_PASS")),
            "version": row.get("version"),
            "task_class": "code_generation",
            "task_description": _prompt(row["repo"], row["problem_statement"]),
            "expected_answer": "",
        })
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps({"tasks": tasks, "meta": {
            "benchmark": "swebench_lite", "hf_id": HF_ID,
            "split": "test", "count": len(tasks),
        }}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    repos = sorted({t["repo"] for t in tasks})
    print(f"Saved {len(tasks)} instances -> {OUT}")
    print(f"Repos ({len(repos)}): {', '.join(repos)}")
    print("\nNext: python scripts/swebench_run.py --limit 5   (needs git; "
          "Docker only for resolution scoring)")


if __name__ == "__main__":
    main()
