#!/usr/bin/env python3
"""Step 1.3 — controlled injection of the rare failure classes.

Short single-tool tasks cannot drift or overflow, so `context_overflow` and
`prompt_drift` are near-absent in collected data. This script manufactures
*semi-synthetic* examples of those two classes by taking REAL collected traces
(real task + real tool calls) and structurally corrupting them:

  context_overflow  — duplicate already-completed steps so the agent visibly
                      repeats itself and burns its turn budget.
  prompt_drift      — insert a distractor sub-goal mid-trace that pulls the
                      agent onto an off-task tool.

These are TRAINING-ONLY augmentations. They are written to data/synthetic/
(the training pool) and every row is tagged `"synthetic": true, "injected":
true` with full provenance. The evaluation sets (RealBench, GAIA human
agreement) are separate and remain 100% real, untouched runs — state this
explicitly in the paper; it is a standard, accepted technique.

Output rows match the DSPy Diagnostician ingestion schema
(scripts/compile_diagnostician.py): task_description, observer_flags,
critic_scores, trace_summary, failure_class, failure_manifestation.

Run:
  python scripts/inject_rare_classes.py --per-class 80
  python scripts/inject_rare_classes.py --per-class 80 --sources gsm8k,hotpotqa,2wiki,gaia
"""
from __future__ import annotations

import json
import random
import sys
import uuid
from pathlib import Path

import click

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

DATA_ROOT = Path("data")
SYNTH_DIR = DATA_ROOT / "synthetic"

# Tools the real executor exposes, split into on/off-task per task class so the
# injected distractor (drift) uses a tool that is genuinely wrong for the task.
_OFF_TASK_TOOL = {
    "reasoning": "web_search",
    "data_analysis": "web_search",
    "code_generation": "web_search",
    "code_or_write": "web_search",
    "web_research": "calculator",
    "general": "web_search",
}
_INTERNAL = {"__llm__", "__tool_use_failed__"}


# ── Source loading ────────────────────────────────────────────────────────────

def _result_files(sources: list[str]) -> list[Path]:
    files: list[Path] = []
    for src in sources:
        rdir = DATA_ROOT / src / "results"
        if rdir.exists():
            files.extend(sorted(rdir.glob("*.json")))
    return files


def _load_base_traces(sources: list[str]) -> list[dict]:
    """Real, clean-ish runs we can corrupt: no run error and ≥1 real tool call."""
    bases = []
    for fp in _result_files(sources):
        try:
            r = json.loads(fp.read_text(encoding="utf-8"))
        except Exception:
            continue
        if r.get("run_error"):
            continue
        trace = r.get("trace") or []
        tool_turns = [e for e in trace if e.get("tool_name") not in _INTERNAL]
        if not tool_turns:
            continue
        bases.append(r)
    return bases


# ── Trace helpers ─────────────────────────────────────────────────────────────

def _entry(turn, tool, args, result, llm) -> dict:
    return {
        "turn": turn, "tool_name": tool, "tool_args": args,
        "tool_result": result, "llm_output": llm,
        "latency_ms": random.randint(80, 400), "token_count": random.randint(20, 80),
    }


def _summarise(trace: list[dict]) -> str:
    lines = []
    for e in trace:
        if e["tool_name"] == "__llm__":
            lines.append(f"turn {e['turn']}: [LLM] {str(e.get('llm_output',''))[:80]}")
        else:
            lines.append(
                f"turn {e['turn']}: {e['tool_name']}("
                f"{json.dumps(e.get('tool_args',{}))[:50]}) -> "
                f"{str(e.get('tool_result',''))[:60]}"
            )
    return "\n".join(lines)


def _critic_scores(correctness, completeness, efficiency, safety=4.5) -> dict:
    overall = round(correctness * 0.4 + completeness * 0.3 + efficiency * 0.2 + safety * 0.1, 3)
    return {
        "correctness": round(correctness, 1), "completeness": round(completeness, 1),
        "efficiency": round(efficiency, 1), "safety": round(safety, 1),
        "overall": overall, "pass_fail": overall >= 3.5,
    }


def _rin(lo, hi) -> float:
    return round(random.uniform(lo, hi), 1)


# ── Injection: context_overflow ─────────────────────────────────────────────

def inject_context_overflow(base: dict) -> dict:
    trace = [dict(e) for e in base["trace"]]
    tool_turns = [e for e in trace if e["tool_name"] not in _INTERNAL]
    # keep the original completed steps, then make the agent loop on the last one
    completed = tool_turns[: max(1, len(tool_turns))]
    new_trace = [dict(e) for e in completed]
    repeat = random.choice(completed)
    n_rep = random.randint(3, 6)
    start = len(new_trace)
    flags = []
    for i in range(n_rep):
        t = start + i
        new_trace.append(_entry(
            t, repeat["tool_name"], repeat.get("tool_args", {}),
            repeat.get("tool_result", ""),
            f"Running {repeat['tool_name']} again — progress seems stalled, retrying.",
        ))
        flags.append({
            "flag_type": "tool_repetition", "signal_value": 0.8, "turn": t,
            "description": (f"Turn {t}: '{repeat['tool_name']}' called with identical args "
                            f"as turn {repeat['turn']}. Agent appears to have lost context."),
        })
    total_turns = len(new_trace)
    if total_turns >= 8:
        flags.append({
            "flag_type": "turn_budget_warning", "signal_value": min(1.0, total_turns / 10),
            "turn": total_turns - 1,
            "description": f"Executor used {total_turns} turns repeating completed work.",
        })
    return _build_row(
        base, new_trace, flags,
        _critic_scores(_rin(1.5, 3.0), _rin(1.2, 2.5), _rin(1.0, 2.0)),
        [round(random.uniform(0.05, 0.20), 3) for _ in new_trace],
        "context_overflow", "step_repetition", total_turns,
    )


# ── Injection: prompt_drift ──────────────────────────────────────────────────

_DISTRACTORS = [
    "Let me also set up logging and a CLI wrapper for this.",
    "While I'm here, I'll research best practices for an unrelated edge case.",
    "Adding a tangential optimisation that wasn't requested.",
    "Let me look up background context on a side topic instead.",
    "I'll also draft documentation and examples beyond the ask.",
]


def inject_prompt_drift(base: dict) -> dict:
    trace = [dict(e) for e in base["trace"]]
    tool_turns = [e for e in trace if e["tool_name"] not in _INTERNAL]
    task_class = base.get("task_class", "general")
    off_tool = _OFF_TASK_TOOL.get(task_class, "web_search")
    manifestation = random.choice(["gradual_drift", "step_drift", "oscillating_drift"])

    # keep the first on-task step, then drift onto an off-task subgoal
    keep = tool_turns[:1] if tool_turns else []
    new_trace = [dict(e) for e in keep]
    n_drift = random.randint(2, 3)
    flags, drift_scores = [], [round(random.uniform(0.05, 0.18), 3) for _ in keep]
    start = len(new_trace)
    for i in range(n_drift):
        t = start + i
        if manifestation == "gradual_drift":
            ds = round(min(0.97, 0.30 + i * 0.18 + random.uniform(0, 0.08)), 3)
        else:
            ds = round(random.uniform(0.55, 0.85), 3)
        drift_scores.append(ds)
        new_trace.append(_entry(
            t, off_tool, {"query": f"tangential subgoal {i}"},
            f"Result from off-task {off_tool} call.",
            random.choice(_DISTRACTORS),
        ))
        if ds > 0.45:
            flags.append({
                "flag_type": "prompt_drift", "signal_value": ds, "turn": t,
                "description": (f"Turn {t}: cosine distance from goal = {ds:.4f}. "
                                f"Agent drifted using {off_tool} (off-task for {task_class})."),
            })
    new_trace.append(_entry(
        len(new_trace), "__llm__", {}, "",
        "FINAL ANSWER: produced an off-task deliverable not matching the original goal.",
    ))
    return _build_row(
        base, new_trace, flags,
        _critic_scores(_rin(1.2, 2.5), _rin(1.2, 2.5), _rin(2.0, 3.5)),
        drift_scores, "prompt_drift", manifestation, len(new_trace),
    )


# ── Row builder (DSPy schema + provenance) ──────────────────────────────────

def _build_row(base, trace, flags, scores, drift_scores,
               failure_class, manifestation, turn_count) -> dict:
    return {
        "id": str(uuid.uuid4()),
        "task_description": base.get("task") or base.get("task_description", ""),
        "task_class": base.get("task_class", "general"),
        "observer_flags": json.dumps(flags),
        "critic_scores": json.dumps(scores),
        "trace_summary": _summarise(trace),
        "drift_scores": drift_scores,
        "executor_turn_count": turn_count,
        "failure_class": failure_class,
        "failure_manifestation": manifestation,
        # provenance — keeps these training-only and auditable
        "synthetic": True,
        "injected": True,
        "injection_type": failure_class,
        "source_benchmark": base.get("benchmark"),
        "source_task_id": base.get("task_id"),
        "source_config": (base.get("run_config") or {}).get("config_id"),
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

@click.command()
@click.option("--per-class", default=80, show_default=True,
              help="Injected examples to generate per rare class")
@click.option("--sources", default="gsm8k,hotpotqa,2wiki,gaia", show_default=True,
              help="Comma-separated benchmarks whose real traces seed the injection")
@click.option("--seed", default=42, show_default=True)
def main(per_class: int, sources: str, seed: int) -> None:
    """Generate injected context_overflow + prompt_drift training examples."""
    random.seed(seed)
    src_list = [s.strip() for s in sources.split(",") if s.strip()]
    bases = _load_base_traces(src_list)
    if not bases:
        print(f"No usable base traces found in {src_list}. Run scripts/bench_run.py first.")
        sys.exit(1)
    print(f"Loaded {len(bases)} real base traces from: {', '.join(src_list)}")

    SYNTH_DIR.mkdir(parents=True, exist_ok=True)
    plan = [
        ("context_overflow", inject_context_overflow, SYNTH_DIR / "context_overflow_injected.jsonl"),
        ("prompt_drift",     inject_prompt_drift,     SYNTH_DIR / "prompt_drift_injected.jsonl"),
    ]
    for cls, fn, out in plan:
        rows = [fn(random.choice(bases)) for _ in range(per_class)]
        with out.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"  {cls:<18} {len(rows):>4} injected examples -> {out}")

    print("\nThese are TRAINING-ONLY. Test sets (RealBench / GAIA) stay 100% real.")
    print("Next: python scripts/compile_diagnostician.py  (or recompile_diagnostician_v2.py)")


if __name__ == "__main__":
    main()
