#!/usr/bin/env python3
"""Generate synthetic failure logs for each of the 5 taxonomy classes.

Outputs one JSONL file per class to data/synthetic/.
Each line is a self-contained training example usable by DSPy BootstrapFewShot.

Run: python scripts/generate_synthetic_data.py [--per-class N]
"""
from __future__ import annotations

import json
import random
import sys
import uuid
from pathlib import Path

import click

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

OUTPUT_DIR = Path("data/synthetic")

TASK_DESCRIPTIONS = {
    "code_generation": [
        "Write a Python function that parses CSV files and returns a list of dicts",
        "Implement a binary search algorithm that works on sorted lists",
        "Create a REST API endpoint that validates email addresses",
        "Write a decorator that caches function results with a TTL",
        "Implement a linked list with insert, delete, and search operations",
    ],
    "web_research": [
        "Search for recent papers on speculative decoding in LLMs",
        "Find the current price of NVIDIA stock and summarise recent news",
        "Research the top 5 Python web frameworks and compare their features",
        "Find information about the latest Claude model releases from Anthropic",
        "Search for tutorials on building multi-agent systems with LangGraph",
    ],
    "data_analysis": [
        "Analyse the sales data and produce a quarterly summary report",
        "Calculate compound interest on $10,000 at 5% annual rate for 3 years",
        "Compute the average, median, and standard deviation of a dataset",
        "Find the top 10 most frequent words in a given text corpus",
        "Determine the correlation between two numeric datasets",
    ],
    "reasoning": [
        "Determine if 2 to the power of 32 is greater than 4 billion",
        "Prove that the square root of 2 is irrational",
        "Explain why a hash map has O(1) average lookup time",
        "Determine the optimal strategy for the Monty Hall problem",
        "Analyse the time complexity of merge sort",
    ],
}

TOOL_NAMES = ["calculator", "web_search", "write_file", "read_file"]


def rand_task() -> tuple[str, str]:
    cls = random.choice(list(TASK_DESCRIPTIONS.keys()))
    return cls, random.choice(TASK_DESCRIPTIONS[cls])


def rand_scores(low: float = 1.5, high: float = 4.5) -> dict:
    c = round(random.uniform(low, high), 1)
    comp = round(random.uniform(low, high), 1)
    eff = round(random.uniform(low, high), 1)
    saf = round(random.uniform(3.0, 5.0), 1)
    overall = round(c * 0.4 + comp * 0.3 + eff * 0.2 + saf * 0.1, 2)
    return {"correctness": c, "completeness": comp, "efficiency": eff, "safety": saf, "overall": overall}


# ── Per-class generators ──────────────────────────────────────────────────────

def gen_prompt_drift(n: int) -> list[dict]:
    examples = []
    for _ in range(n):
        task_cls, task = rand_task()
        drift_scores = [round(random.uniform(0.02, 0.08), 3)] + \
                       [round(random.uniform(0.12, 0.22), 3)] + \
                       [round(random.uniform(0.30, 0.55), 3)] + \
                       [round(random.uniform(0.55, 0.80), 3)]
        flags = [
            {"flag_type": "prompt_drift", "signal_value": s, "turn": i, "description": f"Drift at turn {i}"}
            for i, s in enumerate(drift_scores) if s > 0.45
        ]
        scores = rand_scores(1.5, 3.0)
        trace = [
            {"turn": i, "tool_name": random.choice(TOOL_NAMES), "tool_args": {"query": f"step {i}"},
             "tool_result": "some result", "llm_output": f"executing step {i}", "latency_ms": 200, "token_count": 40}
            for i in range(len(drift_scores))
        ]
        examples.append({
            "id": str(uuid.uuid4()),
            "task_description": task,
            "task_class": task_cls,
            "observer_flags": json.dumps(flags),
            "critic_scores": json.dumps(scores),
            "trace_summary": _summarise_trace(trace),
            "failure_class": "prompt_drift",
            "failure_manifestation": "none",
        })
    return examples


def gen_tool_misuse(n: int) -> list[dict]:
    examples = []
    error_msgs = [
        "Error: unknown tool 'search_web' — did you mean 'web_search'?",
        "Error: argument 'path' expected str, got None",
        "ValidationError: tool 'write_file' requires 'content' field",
        "Error: execute_query called before connect_db",
        "Error: calculator received non-numeric expression '3 + abc'",
    ]
    for _ in range(n):
        task_cls, task = rand_task()
        n_errors = random.randint(1, 3)
        flags = [
            {"flag_type": "tool_error_loop", "signal_value": 0.9, "turn": i,
             "description": f"Tool retried after error at turn {i - 1}"}
            for i in range(1, n_errors + 1)
        ]
        scores = rand_scores(1.0, 2.8)
        wrong_tool = random.choice(["search_web", "execute_query", "write_files", "calc"])
        trace = [
            {"turn": 0, "tool_name": wrong_tool, "tool_args": {}, "tool_result": random.choice(error_msgs),
             "llm_output": f"calling {wrong_tool}…", "latency_ms": 150, "token_count": 35},
            {"turn": 1, "tool_name": wrong_tool, "tool_args": {}, "tool_result": random.choice(error_msgs),
             "llm_output": f"retrying {wrong_tool}…", "latency_ms": 150, "token_count": 35},
        ]
        examples.append({
            "id": str(uuid.uuid4()),
            "task_description": task,
            "task_class": task_cls,
            "observer_flags": json.dumps(flags),
            "critic_scores": json.dumps(scores),
            "trace_summary": _summarise_trace(trace),
            "failure_class": "tool_misuse",
            "failure_manifestation": "tool_misuse",
        })
    return examples


def gen_context_overflow(n: int) -> list[dict]:
    examples = []
    for _ in range(n):
        task_cls, task = rand_task()
        n_repeats = random.randint(1, 3)
        flags = [
            {"flag_type": "tool_repetition", "signal_value": 0.8, "turn": i + 2,
             "description": f"Turn {i + 2}: same call as turn {i}"}
            for i in range(n_repeats)
        ]
        turn_budget_flag = {"flag_type": "turn_budget_warning", "signal_value": 0.85, "turn": 9,
                            "description": "Used 85% of turn budget"}
        flags.append(turn_budget_flag)
        scores = rand_scores(1.5, 3.5)
        tool = random.choice(TOOL_NAMES)
        trace = [
            {"turn": i, "tool_name": tool, "tool_args": {"q": "same query"},
             "tool_result": "result", "llm_output": f"doing step {i} again…", "latency_ms": 200, "token_count": 40}
            for i in range(8)
        ]
        examples.append({
            "id": str(uuid.uuid4()),
            "task_description": task,
            "task_class": task_cls,
            "observer_flags": json.dumps(flags),
            "critic_scores": json.dumps(scores),
            "trace_summary": _summarise_trace(trace),
            "failure_class": "context_overflow",
            "failure_manifestation": "none",
        })
    return examples


def gen_goal_misalignment(n: int) -> list[dict]:
    examples = []
    for _ in range(n):
        task_cls, task = rand_task()
        scores = {
            "correctness": round(random.uniform(1.0, 2.5), 1),
            "completeness": round(random.uniform(1.0, 2.5), 1),
            "efficiency": round(random.uniform(4.0, 5.0), 1),
            "safety": round(random.uniform(4.0, 5.0), 1),
            "overall": 2.2,
        }
        flags = []
        trace = [
            {"turn": i, "tool_name": random.choice(TOOL_NAMES), "tool_args": {},
             "tool_result": "proxy result", "llm_output": f"optimising for speed at turn {i}",
             "latency_ms": 50, "token_count": 20}
            for i in range(3)
        ]
        examples.append({
            "id": str(uuid.uuid4()),
            "task_description": task,
            "task_class": task_cls,
            "observer_flags": json.dumps(flags),
            "critic_scores": json.dumps(scores),
            "trace_summary": _summarise_trace(trace),
            "failure_class": "goal_misalignment",
            "failure_manifestation": "none",
        })
    return examples


def gen_hallucination_loop(n: int) -> list[dict]:
    examples = []
    false_claims = [
        "Chen et al. 2024 published 'Autonomous Agent Failure Modes' at NeurIPS",
        "The weather API returned 200 OK with temperature 22°C",
        "revenue = $1.2M (computed from tool result $120K)",
        "Python 4.0 was released in March 2025 with async-first semantics",
        "The endpoint /api/v3/agents exists and returns agent status",
    ]
    for _ in range(n):
        task_cls, task = rand_task()
        scores = rand_scores(1.0, 2.5)
        claim = random.choice(false_claims)
        trace = [
            {"turn": i, "tool_name": "__llm__" if i % 2 else "web_search",
             "tool_args": {} if i % 2 else {"query": "verification"},
             "tool_result": "" if i % 2 else "Error: no results found",
             "llm_output": f"As established: {claim}", "latency_ms": 300, "token_count": 60}
            for i in range(4)
        ]
        flags = []
        examples.append({
            "id": str(uuid.uuid4()),
            "task_description": task,
            "task_class": task_cls,
            "observer_flags": json.dumps(flags),
            "critic_scores": json.dumps(scores),
            "trace_summary": _summarise_trace(trace),
            "failure_class": "hallucination_loop",
            "failure_manifestation": "hallucination_loop",
        })
    return examples


def gen_clean(n: int) -> list[dict]:
    examples = []
    for _ in range(n):
        task_cls, task = rand_task()
        scores = rand_scores(3.5, 5.0)
        scores["overall"] = round(
            scores["correctness"] * 0.4 + scores["completeness"] * 0.3 +
            scores["efficiency"] * 0.2 + scores["safety"] * 0.1, 2
        )
        trace = [
            {"turn": i, "tool_name": random.choice(TOOL_NAMES), "tool_args": {},
             "tool_result": "success", "llm_output": f"step {i} completed", "latency_ms": 200, "token_count": 40}
            for i in range(random.randint(1, 4))
        ]
        examples.append({
            "id": str(uuid.uuid4()),
            "task_description": task,
            "task_class": task_cls,
            "observer_flags": json.dumps([]),
            "critic_scores": json.dumps(scores),
            "trace_summary": _summarise_trace(trace),
            "failure_class": "none",
            "failure_manifestation": "none",
        })
    return examples


def _summarise_trace(trace: list[dict]) -> str:
    lines = []
    for e in trace:
        if e["tool_name"] == "__llm__":
            lines.append(f"turn {e['turn']}: [LLM] {e['llm_output'][:60]}")
        else:
            lines.append(f"turn {e['turn']}: {e['tool_name']} → {e['tool_result'][:40]}")
    return "\n".join(lines)


GENERATORS = {
    "prompt_drift": gen_prompt_drift,
    "tool_misuse": gen_tool_misuse,
    "context_overflow": gen_context_overflow,
    "goal_misalignment": gen_goal_misalignment,
    "hallucination_loop": gen_hallucination_loop,
    "none": gen_clean,
}


@click.command()
@click.option("--per-class", default=100, show_default=True, help="Examples per failure class")
@click.option("--seed", default=42, show_default=True, help="Random seed for reproducibility")
def main(per_class: int, seed: int) -> None:
    """Generate synthetic failure data for all 6 classes (5 failures + clean)."""
    random.seed(seed)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    total = 0
    for cls, gen_fn in GENERATORS.items():
        examples = gen_fn(per_class)
        out_path = OUTPUT_DIR / f"{cls}.jsonl"
        with out_path.open("w", encoding="utf-8") as f:
            for ex in examples:
                f.write(json.dumps(ex) + "\n")
        total += len(examples)
        print(f"  {cls}: {len(examples)} examples → {out_path}")
    print(f"\nTotal: {total} examples written to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
