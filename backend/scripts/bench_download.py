#!/usr/bin/env python3
"""Download task benchmarks into ARIA's common task schema.

Covers the directly-runnable benchmarks (those the 4-tool executor can
actually attempt): GSM8K, HotpotQA, 2WikiMultihopQA. GAIA has its own
downloader (scripts/gaia_download.py --all-levels); SWE-bench Lite and
tau-bench have dedicated adapters because they need extra tooling.

All datasets are pulled from HuggingFace as parquet-native mirrors
(datasets>=4 dropped support for script-based repos like the official
hotpotqa/hotpot_qa and xanhho/2WikiMultihopQA).

Common task schema written to data/<bench>/tasks.json:
  {
    "tasks": [
      {
        "id": "gsm8k_00001",
        "benchmark": "gsm8k",
        "task": "<raw question>",
        "task_description": "<formatted prompt the executor sees>",
        "task_class": "reasoning" | "web_research" | ...,
        "expected_answer": "<gold answer>",
        "level": null,
        "meta": {...}
      },
      ...
    ],
    "meta": {"benchmark": ..., "split": ..., "count": ..., "hf_id": ...}
  }

Run:
  python scripts/bench_download.py gsm8k        --limit 120
  python scripts/bench_download.py hotpotqa     --limit 120
  python scripts/bench_download.py 2wiki        --limit 120
  python scripts/bench_download.py all          --limit 120
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import click

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

DATA_ROOT = Path("data")


# ── Task prompt formatting ──────────────────────────────────────────────────

def _format(question: str) -> str:
    return (
        "Solve the following task step by step. Use available tools as needed.\n"
        "When you have the final answer, state it clearly starting with "
        "\"FINAL ANSWER:\" on its own line.\n\n"
        f"Task: {question}"
    )


# ── Gold-answer extraction per benchmark ────────────────────────────────────

def _gsm8k_answer(raw: str) -> str:
    # GSM8K answers end with "#### <number>"
    m = re.search(r"####\s*(.+)\s*$", raw.strip())
    return (m.group(1) if m else raw).strip().replace(",", "")


def _first(x) -> str:
    if isinstance(x, (list, tuple)):
        return str(x[0]) if x else ""
    return str(x)


# ── Benchmark registry ──────────────────────────────────────────────────────
# Each entry knows how to load and normalise one benchmark.

def _load_gsm8k(limit: int) -> tuple[list[dict], dict]:
    from datasets import load_dataset
    hf_id, split = "openai/gsm8k", "test"
    ds = load_dataset(hf_id, name="main", split=split)
    tasks = []
    for i, row in enumerate(ds):
        if limit and i >= limit:
            break
        q = row["question"].strip()
        tasks.append({
            "id": f"gsm8k_{i:05d}",
            "benchmark": "gsm8k",
            "task": q,
            "task_description": _format(q),
            "task_class": "reasoning",
            "expected_answer": _gsm8k_answer(row["answer"]),
            "level": None,
            "meta": {"rationale": row["answer"]},
        })
    return tasks, {"benchmark": "gsm8k", "split": split, "hf_id": hf_id}


def _load_hotpotqa(limit: int) -> tuple[list[dict], dict]:
    from datasets import load_dataset
    hf_id, split = "lucadiliello/hotpotqa", "validation"
    ds = load_dataset(hf_id, split=split)
    tasks = []
    for i, row in enumerate(ds):
        if limit and i >= limit:
            break
        q = row["question"].strip()
        tasks.append({
            "id": f"hotpotqa_{i:05d}",
            "benchmark": "hotpotqa",
            "task": q,
            "task_description": _format(q),
            "task_class": "web_research",
            "expected_answer": _first(row.get("answers")),
            "level": None,
            "meta": {"key": row.get("key")},
        })
    return tasks, {"benchmark": "hotpotqa", "split": split, "hf_id": hf_id}


def _load_2wiki(limit: int) -> tuple[list[dict], dict]:
    from datasets import load_dataset
    # Parquet-native mirror (cols: _id, type, question, context,
    # supporting_facts, evidences, answer). The official xanhho/kamelliao
    # repos are script-based and dead on datasets>=4.
    hf_id, split = "voidful/2WikiMultihopQA", "validation"
    ds = load_dataset(hf_id, split=split)
    tasks = []
    for i, row in enumerate(ds):
        if limit and i >= limit:
            break
        q = str(row.get("question", "")).strip()
        ans = row.get("answer") or row.get("answers")
        tasks.append({
            "id": f"2wiki_{i:05d}",
            "benchmark": "2wiki",
            "task": q,
            "task_description": _format(q),
            "task_class": "web_research",
            "expected_answer": _first(ans),
            "level": None,
            "meta": {"type": row.get("type")},
        })
    return tasks, {"benchmark": "2wiki", "split": split, "hf_id": hf_id}


LOADERS = {
    "gsm8k": _load_gsm8k,
    "hotpotqa": _load_hotpotqa,
    "2wiki": _load_2wiki,
}


def _download_one(bench: str, limit: int) -> None:
    print(f"\n=== {bench} ===")
    tasks, meta = LOADERS[bench](limit)
    meta["count"] = len(tasks)
    out_dir = DATA_ROOT / bench
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "tasks.json"
    out_path.write_text(
        json.dumps({"tasks": tasks, "meta": meta}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"  Saved {len(tasks)} tasks -> {out_path}")
    print(f"  Sample: {tasks[0]['task'][:80]}...")
    print(f"  Gold:   {tasks[0]['expected_answer']!r}")


@click.command()
@click.argument("benchmark", type=click.Choice(list(LOADERS) + ["all"]))
@click.option("--limit", default=120, show_default=True,
              help="Max tasks to pull (0 = all available)")
def main(benchmark: str, limit: int) -> None:
    """Download a benchmark into data/<bench>/tasks.json."""
    targets = list(LOADERS) if benchmark == "all" else [benchmark]
    for b in targets:
        try:
            _download_one(b, limit)
        except Exception as exc:
            print(f"  ERROR downloading {b}: {type(exc).__name__}: {exc}")
    print("\nNext: python scripts/bench_run.py --benchmark <bench> "
          "--executor-model llama-3.1-8b-instant --max-turns 8")


if __name__ == "__main__":
    main()
