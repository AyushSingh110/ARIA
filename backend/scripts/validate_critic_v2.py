#!/usr/bin/env python3
"""Validate Critic v2 on the 10 gap runs.

Loads each gap result, re-runs ONLY the Critic v2 against the saved
trace + output, and shows whether it now correctly detects the failure.

No API calls to Groq for Executor/Orchestrator — only Critic is called.

Run: python scripts/validate_critic_v2.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

RESULTS_DIR = Path("data/realbench/results")


def run_critic_on_saved_result(result: dict) -> dict:
    """Re-run Critic v2 on a saved result's trace and output."""
    import dspy
    from aria.agents.critic import _build_human_message, _build_scores, _extract_json, _get_llm
    from langchain_core.messages import HumanMessage, SystemMessage
    from aria.agents.critic import _SYSTEM_PROMPT

    task   = result["task"]
    output = result.get("executor_output", "No output.")
    trace  = result.get("executor_trace") or []

    # Reconstruct trace from trace_summary if executor_trace not available
    if not trace and result.get("trace_summary"):
        # Use trace_summary as pseudo-trace for the LLM
        trace = []
        human_msg = (
            f"Task: {task}\n\n"
            f"Execution trace (summary):\n{result['trace_summary']}\n\n"
            f"Final executor output:\n{output[:600]}"
        )
    else:
        human_msg = _build_human_message(task, output, trace)

    llm      = _get_llm()
    response = llm.invoke([SystemMessage(content=_SYSTEM_PROMPT), HumanMessage(content=human_msg)])

    try:
        parsed = _extract_json(response.content)
        scores = _build_scores(parsed)
    except Exception as exc:
        return {"error": str(exc)}

    return {
        "requirements":            scores["requirement_checklist"],
        "satisfied":               scores["requirements_satisfied"],
        "requirement_satisfaction": scores["requirement_satisfaction"],
        "pass_fail":               scores["pass_fail"],
        "correctness":             scores["correctness"],
        "reasoning":               parsed.get("reasoning", ""),
    }


def main():
    gap_files = []
    for f in sorted(RESULTS_DIR.glob("rb_*.json")):
        r = json.loads(f.read_text(encoding="utf-8"))
        if r.get("human_label") == "gap" and not r.get("run_error"):
            gap_files.append((f, r))

    if not gap_files:
        print("No gap-labeled runs found.")
        sys.exit(1)

    print(f"\nCritic v2 Validation — {len(gap_files)} gap runs")
    print("=" * 65)
    print("Expected: Critic v2 should FAIL these runs (pass_fail=False)")
    print("v1 result: Critic v1 PASSED 8/10 of these (gave 5.0 correctness)")
    print("=" * 65)

    v1_passed = 0
    v2_passed = 0
    v2_correctly_failed = 0

    for path, result in gap_files:
        task_id = result["task_id"]
        v1_score = (result.get("critic_scores") or {})
        v1_pf    = v1_score.get("pass_fail", False)
        v1_corr  = v1_score.get("correctness", "?")

        print(f"\n[{task_id}] {result['task'][:70]}")
        print(f"  Human label : gap")
        print(f"  v1 Critic   : correctness={v1_corr}  pass={v1_pf}")

        v2 = run_critic_on_saved_result(result)

        if "error" in v2:
            print(f"  v2 Critic   : ERROR — {v2['error'][:80]}")
            continue

        req_n   = len(v2["requirements"])
        req_sat = sum(v2["satisfied"])
        pf      = v2["pass_fail"]
        rs      = v2["requirement_satisfaction"]

        print(f"  v2 Critic   : req_satisfaction={req_sat}/{req_n} ({rs:.0%})  pass={pf}")
        print(f"  v2 reasoning: {v2['reasoning'][:120]}")
        print("  Requirements:")
        for req, ok in zip(v2["requirements"], v2["satisfied"]):
            mark = "OK  " if ok else "MISS"
            print(f"    [{mark}] {req}")

        if v1_pf:
            v1_passed += 1
        if pf:
            v2_passed += 1
        else:
            v2_correctly_failed += 1

    print("\n" + "=" * 65)
    print(f"SUMMARY ({len(gap_files)} gap runs that humans labeled as failures):")
    print(f"  v1 Critic passed (should have FAILED) : {v1_passed}/{len(gap_files)}")
    print(f"  v2 Critic passed (false positives)    : {v2_passed}/{len(gap_files)}")
    print(f"  v2 Critic correctly FAILED            : {v2_correctly_failed}/{len(gap_files)}")
    print("=" * 65)

    improvement = v1_passed - v2_passed
    if improvement > 0:
        print(f"\nCritic v2 correctly caught {improvement} more failures than v1.")
    elif improvement == 0:
        print("\nNo improvement over v1 — check the prompt or threshold.")
    else:
        print(f"\nv2 introduced {-improvement} new false positives vs v1.")


if __name__ == "__main__":
    main()
