#!/usr/bin/env python3
"""Generate synthetic failure logs grounded in the ARIA Failure Taxonomy v1.

Each example has:
  - task-class-appropriate tool sequences (Task Ontology)
  - meaningful trace text the DSPy Diagnostician can reason over
  - correct failure manifestation subtype labels
  - drift_scores and executor_turn_count for XGBoost feature vector

Outputs one JSONL file per class to data/synthetic/.

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

# ── Task Ontology ─────────────────────────────────────────────────────────────
# Defines ground-truth expected tools per task class.
# "primary" = tools the agent SHOULD use.
# "off_task" = tools that are wrong for this task class (used in drift/misuse).

TASK_ONTOLOGY: dict[str, dict] = {
    "code_generation": {
        "primary": ["write_file", "read_file"],
        "off_task": ["web_search", "calculator"],
        "result_templates": {
            "write_file": "File written: {filename}.py ({lines} lines)",
            "read_file": "File content: def {fn}({args}): ...",
        },
    },
    "web_research": {
        "primary": ["web_search"],
        "secondary": ["write_file"],
        "off_task": ["calculator", "read_file"],
        "result_templates": {
            "web_search": "Found {n} results: {snippet}",
            "write_file": "Summary saved to research_notes.txt",
        },
    },
    "data_analysis": {
        "primary": ["calculator"],
        "secondary": ["read_file"],
        "off_task": ["web_search", "write_file"],
        "result_templates": {
            "calculator": "{expr} = {result}",
            "read_file": "Dataset loaded: {n} rows, columns=[{cols}]",
        },
    },
    "reasoning": {
        "primary": ["calculator"],
        "off_task": ["web_search", "write_file", "read_file"],
        "result_templates": {
            "calculator": "{expr} = {result}",
        },
    },
}

TASK_DESCRIPTIONS: dict[str, list[str]] = {
    "code_generation": [
        "Write a Python function that parses CSV files and returns a list of dicts",
        "Implement a binary search algorithm that works on sorted lists",
        "Create a REST API endpoint that validates email addresses",
        "Write a decorator that caches function results with a TTL",
        "Implement a linked list with insert, delete, and search operations",
        "Write a Python context manager for safe file operations",
        "Implement a rate-limiter class with a sliding window algorithm",
        "Create a function that deep-merges two nested dictionaries",
    ],
    "web_research": [
        "Search for recent papers on speculative decoding in LLMs",
        "Find the latest Claude model releases from Anthropic and summarise changes",
        "Research the top 5 Python web frameworks and compare their features",
        "Search for tutorials on building multi-agent systems with LangGraph",
        "Find information about the current state of AI agent benchmarks",
        "Search for recent advances in retrieval-augmented generation",
        "Find and summarise three sources on transformer attention optimisation",
        "Research open-source alternatives to OpenAI's function calling API",
    ],
    "data_analysis": [
        "Calculate compound interest on $10,000 at 5% annual rate for 3 years",
        "Compute the average, median, and standard deviation of this dataset",
        "Determine if quarterly revenue grew faster than 10% year-over-year",
        "Calculate the break-even point given fixed costs of $50,000 and margin of 40%",
        "Compute the Sharpe ratio for a portfolio with 12% return and 8% volatility",
        "Determine the sample size needed for 95% confidence with 5% margin of error",
    ],
    "reasoning": [
        "Determine if 2 to the power of 32 is greater than 4 billion",
        "Prove that the square root of 2 is irrational",
        "Determine the optimal strategy for the Monty Hall problem",
        "Analyse the time complexity of merge sort and justify the answer",
        "Determine how many prime numbers exist below 100",
        "Calculate whether a 15% tip on a $47.50 bill rounds to $7 or $8",
    ],
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def rand_task() -> tuple[str, str]:
    cls = random.choice(list(TASK_DESCRIPTIONS.keys()))
    return cls, random.choice(TASK_DESCRIPTIONS[cls])


def _fill_result(template: str, task_cls: str) -> str:
    fills = {
        "filename": random.choice(["parser", "binary_search", "cache", "api_handler"]),
        "lines": random.randint(15, 60),
        "fn": random.choice(["parse_csv", "binary_search", "validate_email", "cache_result"]),
        "args": random.choice(["data: list", "arr: list, target: int", "email: str"]),
        "n": random.randint(3, 12),
        "snippet": random.choice([
            "LangGraph v0.2 adds native streaming support",
            "Anthropic releases Claude 3.5 Haiku with improved tool use",
            "New benchmark shows GPT-4o at 72% on AgentBench",
            "DSPy 2.4 improves BootstrapFewShot stability",
        ]),
        "expr": random.choice(["10000 * (1.05**3)", "2**32", "sqrt(144) + 2**8", "47.5 * 0.15"]),
        "result": random.choice(["11576.25", "4294967296", "268.0", "7.125"]),
        "n2": random.randint(100, 5000),
        "cols": "date, revenue, cost",
    }
    fills["n2"] = fills["n"]
    try:
        return template.format(**fills)
    except KeyError:
        return template


def _make_trace_entry(
    turn: int,
    tool: str,
    args: dict,
    result: str,
    llm_output: str,
) -> dict:
    return {
        "turn": turn,
        "tool_name": tool,
        "tool_args": args,
        "tool_result": result,
        "llm_output": llm_output,
        "latency_ms": random.randint(80, 400),
        "token_count": random.randint(20, 80),
    }


def _summarise_trace(trace: list[dict]) -> str:
    lines = []
    for e in trace:
        if e["tool_name"] == "__llm__":
            lines.append(f"turn {e['turn']}: [LLM] {e['llm_output'][:80]}")
        else:
            lines.append(
                f"turn {e['turn']}: {e['tool_name']}({json.dumps(e['tool_args'])[:50]}) "
                f"-> {e['tool_result'][:60]}"
            )
    return "\n".join(lines)


def _critic_scores(
    correctness: float,
    completeness: float,
    efficiency: float,
    safety: float = 4.5,
) -> dict:
    overall = round(correctness * 0.4 + completeness * 0.3 + efficiency * 0.2 + safety * 0.1, 3)
    return {
        "correctness": round(correctness, 1),
        "completeness": round(completeness, 1),
        "efficiency": round(efficiency, 1),
        "safety": round(safety, 1),
        "overall": overall,
        "pass_fail": overall >= 3.5,
    }


def _rand_in(lo: float, hi: float) -> float:
    return round(random.uniform(lo, hi), 1)


# ── Per-class generators ──────────────────────────────────────────────────────

def gen_none(n: int) -> list[dict]:
    """Clean runs: task-appropriate tools, successful results, 1–4 turns."""
    examples = []
    for _ in range(n):
        task_cls, task = rand_task()
        ontology = TASK_ONTOLOGY[task_cls]
        primary = ontology["primary"]
        results_tpl = ontology["result_templates"]

        n_turns = random.randint(1, 4)
        trace = []
        for i in range(n_turns):
            tool = primary[i % len(primary)]
            result = _fill_result(results_tpl.get(tool, "Success."), task_cls)
            trace.append(_make_trace_entry(
                turn=i,
                tool=tool,
                args={"query": f"step {i + 1}"},
                result=result,
                llm_output=f"Step {i + 1} complete. {result[:40]}",
            ))

        drift_scores = [round(random.uniform(0.02, 0.15), 3) for _ in trace]

        examples.append(_build_example(
            task_cls=task_cls,
            task=task,
            trace=trace,
            flags=[],
            scores=_critic_scores(
                _rand_in(3.5, 5.0),
                _rand_in(3.5, 5.0),
                _rand_in(3.5, 5.0),
            ),
            drift_scores=drift_scores,
            failure_class="none",
            manifestation="none",
        ))
    return examples


def gen_prompt_drift(n: int) -> list[dict]:
    """Drift: starts on-task, switches to wrong tools after turn k."""
    examples = []
    manifestation_choices = ["gradual_drift", "step_drift", "oscillating_drift"]

    for _ in range(n):
        task_cls, task = rand_task()
        ontology = TASK_ONTOLOGY[task_cls]
        primary = ontology["primary"]
        off_task = ontology["off_task"]
        results_tpl = ontology["result_templates"]
        manifestation = random.choice(manifestation_choices)

        drift_start = random.randint(1, 2)
        n_turns = random.randint(4, 6)
        trace = []
        drift_scores = []
        flags = []

        for i in range(n_turns):
            if i < drift_start:
                # On-task turns
                tool = primary[0]
                result = _fill_result(results_tpl.get(tool, "Success."), task_cls)
                llm_out = f"Working on the original task at turn {i}."
                drift_score = round(random.uniform(0.05, 0.18), 3)
            elif manifestation == "oscillating_drift":
                # Alternates between correct and off-task
                if i % 2 == 0:
                    tool = primary[0]
                    result = _fill_result(results_tpl.get(tool, "Success."), task_cls)
                    drift_score = round(random.uniform(0.08, 0.25), 3)
                    llm_out = "Returning to original task briefly."
                else:
                    tool = random.choice(off_task)
                    result = f"Result from unrelated {tool} call."
                    drift_score = round(random.uniform(0.50, 0.75), 3)
                    llm_out = f"Exploring {tool} for tangential information."
            else:
                # Drifted turns
                tool = random.choice(off_task)
                result = f"Result from unrelated {tool} call."
                if manifestation == "gradual_drift":
                    drift_score = round(
                        0.20 + (i - drift_start) * 0.12 + random.uniform(0, 0.08), 3
                    )
                else:
                    drift_score = round(random.uniform(0.55, 0.82), 3)
                llm_out = f"Pursuing tangential goal at turn {i} using {tool}."

            drift_scores.append(min(drift_score, 0.99))

            if drift_score > 0.45:
                flags.append({
                    "flag_type": "prompt_drift",
                    "signal_value": round(drift_score, 4),
                    "turn": i,
                    "description": (
                        f"Turn {i}: cosine distance from goal = {drift_score:.4f}. "
                        f"Agent drifted using {tool} (off-task for {task_cls})."
                    ),
                })

            trace.append(_make_trace_entry(
                turn=i,
                tool=tool,
                args={"query": f"step {i}"},
                result=result,
                llm_output=llm_out,
            ))

        examples.append(_build_example(
            task_cls=task_cls,
            task=task,
            trace=trace,
            flags=flags,
            scores=_critic_scores(
                _rand_in(1.2, 2.5),
                _rand_in(1.2, 2.5),
                _rand_in(2.0, 3.5),
            ),
            drift_scores=drift_scores,
            failure_class="prompt_drift",
            manifestation=manifestation,
        ))
    return examples


def gen_tool_misuse(n: int) -> list[dict]:
    """Tool misuse: wrong tool name, bad args, or dependency violation."""
    examples = []
    manifestation_choices = ["unknown_tool", "wrong_args", "wrong_sequence", "wrong_tool"]

    # Maps task class to its wrong-name variants
    wrong_tool_names = {
        "write_file": ["write_files", "save_file", "file_write"],
        "web_search": ["search_web", "google_search", "browser_search"],
        "calculator": ["calc", "compute", "math_tool"],
        "read_file": ["read_files", "file_read", "load_file"],
    }

    for _ in range(n):
        task_cls, task = rand_task()
        ontology = TASK_ONTOLOGY[task_cls]
        primary_tool = ontology["primary"][0]
        manifestation = random.choice(manifestation_choices)

        trace = []
        flags = []
        n_errors = random.randint(1, 3)

        if manifestation == "unknown_tool":
            wrong_name = random.choice(wrong_tool_names.get(primary_tool, ["unknown_tool"]))
            for i in range(n_errors + 1):
                error_msg = (
                    f"Error: unknown tool '{wrong_name}' - "
                    f"did you mean '{primary_tool}'?"
                )
                trace.append(_make_trace_entry(
                    turn=i,
                    tool=wrong_name,
                    args={"query": f"attempt {i + 1}"},
                    result=error_msg,
                    llm_output=f"Calling {wrong_name} to complete task.",
                ))
                if i > 0:
                    flags.append({
                        "flag_type": "tool_error_loop",
                        "signal_value": 0.9,
                        "turn": i,
                        "description": (
                            f"Turn {i}: '{wrong_name}' retried after error on turn {i - 1}."
                        ),
                    })

        elif manifestation == "wrong_args":
            bad_arg_errors = {
                "calculator": "Error: calculator received non-numeric expression '{bad_expr}'",
                "write_file": "ValidationError: 'content' field required, got None",
                "read_file": "Error: argument 'path' expected str, got NoneType",
                "web_search": "ValidationError: 'query' must be a non-empty string",
            }
            err = bad_arg_errors.get(primary_tool, "Error: invalid arguments")
            for i in range(n_errors + 1):
                trace.append(_make_trace_entry(
                    turn=i,
                    tool=primary_tool,
                    args={},
                    result=err.format(bad_expr=random.choice(["3 + abc", "null", "undefined"])),
                    llm_output=f"Attempting {primary_tool} with arguments.",
                ))
                if i > 0:
                    flags.append({
                        "flag_type": "tool_error_loop",
                        "signal_value": 0.9,
                        "turn": i,
                        "description": (
                            f"Turn {i}: '{primary_tool}' retried after validation error on turn {i - 1}."
                        ),
                    })

        elif manifestation == "wrong_sequence":
            # e.g., write_file before read_file when reading is required
            trace.append(_make_trace_entry(
                turn=0,
                tool="write_file",
                args={"path": "output.txt"},
                result="Error: cannot write - source file not yet read",
                llm_output="Writing output before reading source data.",
            ))
            trace.append(_make_trace_entry(
                turn=1,
                tool="write_file",
                args={"path": "output.txt"},
                result="Error: cannot write - source file not yet read",
                llm_output="Retrying write without reading source first.",
            ))
            flags.append({
                "flag_type": "tool_error_loop",
                "signal_value": 0.9,
                "turn": 1,
                "description": "Turn 1: 'write_file' retried after error on turn 0 (wrong sequence).",
            })

        else:  # wrong_tool
            off_task = ontology["off_task"]
            wrong = random.choice(off_task)
            for i in range(2):
                trace.append(_make_trace_entry(
                    turn=i,
                    tool=wrong,
                    args={"query": f"using wrong tool at turn {i}"},
                    result=f"Result from {wrong} (wrong tool for this task type)",
                    llm_output=f"Using {wrong} instead of {primary_tool}.",
                ))

        drift_scores = [round(random.uniform(0.05, 0.20), 3) for _ in trace]

        examples.append(_build_example(
            task_cls=task_cls,
            task=task,
            trace=trace,
            flags=flags,
            scores=_critic_scores(
                _rand_in(1.0, 2.5),
                _rand_in(1.0, 2.2),
                _rand_in(1.0, 2.5),
            ),
            drift_scores=drift_scores,
            failure_class="tool_misuse",
            manifestation=manifestation,
        ))
    return examples


def gen_context_overflow(n: int) -> list[dict]:
    """Context overflow: agent repeats completed steps, hits turn budget."""
    examples = []
    manifestation_choices = ["step_repetition", "exhaustion", "constraint_violation"]

    for _ in range(n):
        task_cls, task = rand_task()
        ontology = TASK_ONTOLOGY[task_cls]
        primary_tool = ontology["primary"][0]
        results_tpl = ontology["result_templates"]
        manifestation = random.choice(manifestation_choices)

        correct_result = _fill_result(results_tpl.get(primary_tool, "Success."), task_cls)
        args = {"query": "initial task query"}
        n_turns = random.randint(6, 10)
        trace = []
        flags = []

        # First 2 turns: correct execution
        for i in range(2):
            trace.append(_make_trace_entry(
                turn=i,
                tool=primary_tool,
                args=args,
                result=correct_result,
                llm_output=f"Completed step {i + 1}. {correct_result[:40]}",
            ))

        # Remaining turns: repeat the same call (agent forgot it already ran)
        for i in range(2, n_turns):
            trace.append(_make_trace_entry(
                turn=i,
                tool=primary_tool,
                args=args,
                result=correct_result,
                llm_output=f"Running {primary_tool} again - seems like progress stalled.",
            ))
            flags.append({
                "flag_type": "tool_repetition",
                "signal_value": 0.8,
                "turn": i,
                "description": (
                    f"Turn {i}: '{primary_tool}' called with identical args as turn 0. "
                    f"Agent appears to have lost context."
                ),
            })

        # Add turn budget warning if close to limit
        if n_turns >= 8:
            ratio = round(n_turns / 10, 2)
            flags.append({
                "flag_type": "turn_budget_warning",
                "signal_value": ratio,
                "turn": n_turns - 1,
                "description": (
                    f"Executor used {n_turns}/10 turns ({int(ratio * 100)}%). "
                    f"Agent consumed most of its turn budget."
                ),
            })

        drift_scores = [round(random.uniform(0.05, 0.20), 3) for _ in trace]

        examples.append(_build_example(
            task_cls=task_cls,
            task=task,
            trace=trace,
            flags=flags,
            scores=_critic_scores(
                _rand_in(1.5, 3.0),
                _rand_in(1.2, 2.5),
                _rand_in(1.0, 2.2),
            ),
            drift_scores=drift_scores,
            failure_class="context_overflow",
            manifestation=manifestation,
            turn_count=n_turns,
        ))
    return examples


def gen_goal_misalignment(n: int) -> list[dict]:
    """Goal misalignment: clean trace, correct tools, but wrong objective solved.

    Every trace is task-consistent: the wrong approach matches the actual task.
    No observer flags. Critic scores low because agent solved a proxy task.
    """
    examples = []
    manifestation_choices = ["partial_completion", "proxy_optimization", "specification_miss"]

    # Each entry: (keyword_in_task, task_class, tool, wrong_arg_or_content,
    #              tool_result, final_answer_explaining_what_went_wrong)
    TASK_WRONG_APPROACHES = [
        # data_analysis
        ("compound interest", "data_analysis", "calculator",
         "10000 * 0.05 * 3", "10000 * 0.05 * 3 = 1500.0",
         "Interest = $1,500. (Used simple interest P*r*t, not compound P*(1+r)^n.)"),
        ("break-even", "data_analysis", "calculator",
         "50000 / 0.4", "50000 / 0.4 = 125000.0",
         "Revenue needed = $125,000. (Computed revenue target, not unit break-even point.)"),
        ("average, median", "data_analysis", "calculator",
         "sum_values / count", "sum_values / count = 42.3",
         "Mean = 42.3. (Computed mean only; median and standard deviation not calculated.)"),
        ("Sharpe ratio", "data_analysis", "calculator",
         "0.12 / 0.08", "0.12 / 0.08 = 1.5",
         "Ratio = 1.5. (Divided return by volatility, omitted risk-free rate adjustment.)"),
        ("sample size", "data_analysis", "calculator",
         "1.96**2 / 0.05**2", "1.96**2 / 0.05**2 = 1536.64",
         "Partial result = 1537. (Used incomplete formula; missing population proportion p*(1-p).)"),
        ("correlation", "data_analysis", "calculator",
         "0.8 - 0.3", "0.8 - 0.3 = 0.5",
         "Difference = 0.5. (Computed difference, not Pearson correlation coefficient.)"),
        # reasoning
        ("2 to the power of 32", "reasoning", "calculator",
         "2 * 32", "2 * 32 = 64",
         "Result = 64. (Multiplied instead of exponentiating; 2**32 = 4,294,967,296.)"),
        ("Monty Hall", "reasoning", "calculator",
         "1/3", "1/3 = 0.333",
         "P = 0.333. (Computed initial door probability, not the switching advantage of 2/3.)"),
        ("prime numbers", "reasoning", "calculator",
         "100 / 4", "100 / 4 = 25.0",
         "Estimate = 25. (Used rough approximation, not enumeration; actual answer is 25 primes.)"),
        ("15% tip", "reasoning", "calculator",
         "47.50 * 0.15", "47.50 * 0.15 = 7.125",
         "Tip = $7.125. (Computed exact value but did not determine which whole dollar it rounds to.)"),
        ("merge sort", "reasoning", "calculator",
         "10 * 1", "10 * 1 = 10",
         "Result = 10. (Used n*1 instead of n*log(n); did not justify the log factor.)"),
        ("square root of 2", "reasoning", "calculator",
         "14142 / 10000", "14142 / 10000 = 1.4142",
         "Decimal = 1.4142. (Computed approximation; task requires an irrationality proof, not a value.)"),
        # code_generation
        ("binary search", "code_generation", "write_file",
         "def search(lst, t): return lst.index(t)", "File written: search.py",
         "Search function written. (Uses list.index O(n), not binary search O(log n).)"),
        ("linked list", "code_generation", "write_file",
         "class Node:\n    def __init__(self, val): self.val = val; self.next = None",
         "File written: linked_list.py",
         "Node class written. (Defines structure only; insert, delete, search methods missing.)"),
        ("CSV", "code_generation", "write_file",
         "def parse(f): return [l.strip().split(',') for l in open(f)]",
         "File written: parser.py",
         "Parser returns list of lists. (Task requires list of dicts keyed by header names.)"),
        ("decorator", "code_generation", "write_file",
         "def cache(fn):\n    memo = {}\n    return lambda *a: memo.setdefault(a, fn(*a))",
         "File written: cache.py",
         "Memoization cache written. (Caches results permanently; no TTL expiry implemented.)"),
        ("rate-limiter", "code_generation", "write_file",
         "class RateLimiter:\n    def __init__(self, rate): self.rate = rate",
         "File written: limiter.py",
         "Skeleton class written. (Stores rate only; sliding window algorithm not implemented.)"),
        ("REST API", "code_generation", "write_file",
         "def validate(email): return '@' in email",
         "File written: validator.py",
         "Basic validator written. (Checks only for '@'; no domain or format validation.)"),
        ("context manager", "code_generation", "write_file",
         "def open_file(path): return open(path)",
         "File written: file_ops.py",
         "File opener written. (Plain function; not a context manager, no __enter__/__exit__.)"),
        ("deep-merge", "code_generation", "write_file",
         "def merge(a, b): return {**a, **b}",
         "File written: merge.py",
         "Shallow merge written. ({**a, **b} overwrites nested keys; not a recursive deep merge.)"),
        ("None and empty", "code_generation", "write_file",
         "def f(x): return x is None",
         "File written: handler.py (4 lines)",
         "Function written. (Handles None only; empty string case not handled.)"),
        # web_research
        ("THREE", "web_research", "web_search",
         "transformer attention mechanism",
         "Found 8 results: Attention is All You Need (2017)...",
         "FINAL ANSWER: Attention uses Q/K/V matrices (Vaswani et al.). (Only one source summarised; task requires three.)"),
        ("top 5", "web_research", "web_search",
         "Python web frameworks comparison 2025",
         "Found 9 results: Django, Flask, FastAPI listed as top frameworks.",
         "FINAL ANSWER: Top frameworks are Django, Flask, FastAPI. (Listed 3 only; task requires 5 with feature comparison.)"),
        ("latest", "web_research", "web_search",
         "Claude model releases Anthropic 2025",
         "Found 6 results: Claude 3.5 Sonnet released Oct 2024.",
         "FINAL ANSWER: Latest release is Claude 3.5 Sonnet. (Did not summarise all releases or changes as required.)"),
        ("speculative decoding", "web_research", "web_search",
         "speculative decoding LLMs papers 2024",
         "Found 11 results: Leviathan et al. 2023, Chen et al. 2023.",
         "FINAL ANSWER: Speculative decoding uses a draft model to propose tokens. (Described mechanism only; missed recent advances.)"),
        ("LangGraph", "web_research", "web_search",
         "LangGraph multi-agent tutorials 2025",
         "Found 7 results: LangGraph quickstart tutorial, official docs.",
         "FINAL ANSWER: LangGraph uses a StateGraph with nodes and edges. (Summarised one tutorial; did not compare multiple sources.)"),
        ("open-source alternatives", "web_research", "web_search",
         "open source function calling API alternatives",
         "Found 8 results: LangChain tools, Llama function calling.",
         "FINAL ANSWER: LangChain tools provide function calling. (Listed one alternative; task requires research across multiple options.)"),
    ]

    def _find_approach(task: str) -> tuple | None:
        task_lower = task.lower()
        for entry in TASK_WRONG_APPROACHES:
            if entry[0].lower() in task_lower:
                return entry
        return None

    for _ in range(n):
        task_cls, task = rand_task()
        manifestation = random.choice(manifestation_choices)

        approach = _find_approach(task)

        if approach:
            _, matched_cls, tool, wrong_arg, tool_result, final_answer = approach
            trace_summary = (
                f"turn 0: {tool}({{'content': '{wrong_arg[:60]}'}}) -> {tool_result}\n"
                f"turn 1: [LLM] FINAL ANSWER: {final_answer}"
            )
            drift_scores = [round(random.uniform(0.05, 0.15), 3),
                            round(random.uniform(0.08, 0.18), 3)]
            turn_count = 2
            task_cls = matched_cls
        else:
            # Fallback: build a task-appropriate but incomplete trace
            ontology = TASK_ONTOLOGY[task_cls]
            primary_tool = ontology["primary"][0]
            results_tpl = ontology["result_templates"]
            result = _fill_result(results_tpl.get(primary_tool, "Success."), task_cls)
            trace_summary = (
                f"turn 0: {primary_tool}({{'query': 'partial attempt'}}) -> {result}\n"
                f"turn 1: [LLM] FINAL ANSWER: Partial result obtained. "
                f"(Agent completed only part of the required task.)"
            )
            drift_scores = [round(random.uniform(0.05, 0.15), 3),
                            round(random.uniform(0.08, 0.18), 3)]
            turn_count = 2

        ex = _build_example(
            task_cls=task_cls,
            task=task,
            trace=[],
            flags=[],
            scores=_critic_scores(
                _rand_in(1.2, 2.5),
                _rand_in(1.0, 2.5),
                _rand_in(3.5, 5.0),
                safety=_rand_in(4.0, 5.0),
            ),
            drift_scores=drift_scores,
            failure_class="goal_misalignment",
            manifestation=manifestation,
            turn_count=turn_count,
        )
        ex["trace_summary"] = trace_summary
        examples.append(ex)
    return examples


def gen_hallucination_loop(n: int) -> list[dict]:
    """Hallucination loop: mostly LLM turns, agent asserts unverified facts."""
    examples = []
    manifestation_choices = [
        "pure_hallucination",
        "tool_bypass",
        "confidence_reinforcement",
    ]

    # Claims must be ABOUT the task topic but factually wrong.
    # This prevents the trace from looking like goal_misalignment (off-topic drift).
    false_claims_by_task = {
        "code_generation": [
            "Binary search on a Python list is O(1) because Python lists are hash-indexed",
            "Python's list.index() uses binary search internally on sorted lists",
            "A linked list in Python requires the ctypes module for pointer management",
            "Python's @functools.ttl_cache provides built-in TTL decorator support",
            "csv.DictReader handles None and empty string fields identically by default",
            "Context managers are only required for file I/O, not other resource types",
            "The dict | operator in Python performs a recursive deep merge of nested dicts",
            "A sliding window rate limiter is built into Python's asyncio.Semaphore class",
        ],
        "web_research": [
            "LangGraph's official benchmark shows 0% agent failure rate on tool-use tasks",
            "Anthropic confirmed Claude 3.5 achieves 98% recall on multi-step agent benchmarks",
            "The 2024 speculative decoding paper demonstrates exactly 10x inference speedup",
            "All major RAG frameworks now consistently outperform fine-tuned models on domain tasks",
            "OpenAI's function calling specification was released as an open standard in 2025",
            "AgentBench v2 shows GPT-4o completes 94% of multi-hop web research tasks correctly",
            "The original attention paper has been replicated with zero errors by all major labs",
            "All Python web frameworks now natively support LLM function-calling interfaces",
        ],
        "data_analysis": [
            "Compound interest on $10,000 at 5% for 3 years is exactly $11,500.00",
            "Standard deviation equals variance divided by sample size for normal distributions",
            "Break-even for $50,000 fixed costs at 40% margin requires exactly 80,000 units",
            "The Sharpe ratio formula does not require the risk-free rate when volatility exceeds 5%",
            "A sample of 200 always achieves 95% confidence regardless of margin of error",
            "A correlation above 0.7 between two variables confirms a direct causal relationship",
            "Year-over-year growth rate is computed by dividing Q4 revenue by Q1 revenue directly",
            "The 95% confidence interval is always the mean plus or minus 2 standard deviations",
        ],
        "reasoning": [
            "2 to the power of 32 equals exactly 4,000,000,000 (four billion)",
            "The square root of 2 equals 1.41421356 exactly as a terminating decimal",
            "In the Monty Hall problem, switching and staying both have 50% probability",
            "Merge sort achieves O(n) average time complexity because each pass is linear",
            "There are exactly 21 prime numbers below 100",
            "15% of $47.50 rounds to exactly $7.00 under standard rounding rules",
            "A hash map has O(log n) lookup because keys are stored in a sorted tree",
            "The irrationality of sqrt(2) was first proved by Euclid in 300 BC exactly",
        ],
    }

    for _ in range(n):
        task_cls, task = rand_task()
        manifestation = random.choice(manifestation_choices)
        claims = false_claims_by_task.get(task_cls, false_claims_by_task["reasoning"])
        false_claim = random.choice(claims)

        trace = []
        n_turns = random.randint(3, 5)

        if manifestation == "pure_hallucination":
            # All LLM turns, zero tool calls
            for i in range(n_turns):
                trace.append(_make_trace_entry(
                    turn=i,
                    tool="__llm__",
                    args={},
                    result="",
                    llm_output=(
                        f"As I know from my training: {false_claim}. "
                        f"This is well-established in the literature."
                        if i == 0
                        else f"To confirm: {false_claim}. No additional verification needed."
                    ),
                ))

        elif manifestation == "tool_bypass":
            # Makes one tool call that fails, then continues inventing
            tool = TASK_ONTOLOGY[task_cls]["primary"][0]
            trace.append(_make_trace_entry(
                turn=0,
                tool=tool,
                args={"query": "verification attempt"},
                result="Error: service unavailable",
                llm_output="Attempting tool verification.",
            ))
            for i in range(1, n_turns):
                trace.append(_make_trace_entry(
                    turn=i,
                    tool="__llm__",
                    args={},
                    result="",
                    llm_output=(
                        f"Tool failed but I can answer from knowledge: {false_claim}."
                        if i == 1
                        else f"Proceeding with confidence: {false_claim}."
                    ),
                ))

        else:  # confidence_reinforcement
            # Each turn reinforces the false claim with more certainty
            certainty = ["I believe", "I am fairly confident", "I am certain", "It is definitively established"]
            for i in range(n_turns):
                cert = certainty[min(i, len(certainty) - 1)]
                trace.append(_make_trace_entry(
                    turn=i,
                    tool="__llm__",
                    args={},
                    result="",
                    llm_output=f"{cert} that {false_claim}.",
                ))

        drift_scores = [round(random.uniform(0.05, 0.20), 3) for _ in trace]

        examples.append(_build_example(
            task_cls=task_cls,
            task=task,
            trace=trace,
            flags=[],
            scores=_critic_scores(
                _rand_in(1.0, 2.2),  # correctness very low
                _rand_in(2.0, 3.5),
                _rand_in(3.0, 5.0),
                safety=_rand_in(3.5, 5.0),
            ),
            drift_scores=drift_scores,
            failure_class="hallucination_loop",
            manifestation=manifestation,
        ))
    return examples


# ── Record builder ────────────────────────────────────────────────────────────

def _build_example(
    task_cls: str,
    task: str,
    trace: list[dict],
    flags: list[dict],
    scores: dict,
    drift_scores: list[float],
    failure_class: str,
    manifestation: str,
    turn_count: int | None = None,
) -> dict:
    return {
        "id": str(uuid.uuid4()),
        "task_description": task,
        "task_class": task_cls,
        "observer_flags": json.dumps(flags),
        "critic_scores": json.dumps(scores),
        "trace_summary": _summarise_trace(trace),
        "drift_scores": drift_scores,
        "executor_turn_count": turn_count if turn_count is not None else len(trace),
        "failure_class": failure_class,
        "failure_manifestation": manifestation,
    }


# ── Registry ──────────────────────────────────────────────────────────────────

GENERATORS = {
    "none": gen_none,
    "prompt_drift": gen_prompt_drift,
    "tool_misuse": gen_tool_misuse,
    "context_overflow": gen_context_overflow,
    "goal_misalignment": gen_goal_misalignment,
    "hallucination_loop": gen_hallucination_loop,
}


@click.command()
@click.option(
    "--per-class",
    default=200,
    show_default=True,
    help="Examples per failure class (200 recommended for stable training)",
)
@click.option("--seed", default=42, show_default=True, help="Random seed for reproducibility")
def main(per_class: int, seed: int) -> None:
    """Generate ARIA-Bench synthetic data grounded in Failure Taxonomy v1."""
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
        print(f"  {cls:<22} {len(examples):>4} examples  ->  {out_path}")

    print(f"\nTotal: {total} examples written to {OUTPUT_DIR}/")
    print(
        "\nNext: python scripts/compile_diagnostician.py "
        "--per-class-limit 160 --val-ratio 0.2"
    )


if __name__ == "__main__":
    main()
