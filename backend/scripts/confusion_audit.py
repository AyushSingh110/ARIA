#!/usr/bin/env python3
"""Confusion Audit -- Research Cycle 0.5

Prints side-by-side examples from the three hard class pairs:
  1. prompt_drift     vs  goal_misalignment
  2. context_overflow vs  tool_misuse
  3. hallucination_loop vs goal_misalignment

For each pair: shows 5 examples per class.
Purpose: verify that class boundaries are humanly distinguishable
before any classifier training.

Outputs a readable report to research/confusion_audit_v1.md

Run: python scripts/confusion_audit.py
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

DATA_DIR   = Path("data/synthetic")
REPORT_DIR = Path("../research")
N_SAMPLES  = 5
SEED       = 42

HARD_PAIRS = [
    ("prompt_drift",      "goal_misalignment",  "Pair A"),
    ("context_overflow",  "tool_misuse",        "Pair B"),
    ("hallucination_loop","goal_misalignment",  "Pair C"),
]


def load_samples(cls: str, n: int, seed: int) -> list[dict]:
    path = DATA_DIR / f"{cls}.jsonl"
    if not path.exists():
        print(f"ERROR: {path} not found. Run generate_synthetic_data.py first.")
        sys.exit(1)
    examples = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    random.seed(seed)
    return random.sample(examples, min(n, len(examples)))


def fmt_example(ex: dict, idx: int) -> str:
    flags  = json.loads(ex["observer_flags"])
    scores = json.loads(ex["critic_scores"])
    flag_str = ", ".join(f["flag_type"] for f in flags) if flags else "none"

    lines = [
        f"  Example {idx + 1}",
        f"  Task class  : {ex['task_class']}",
        f"  Task        : {ex['task_description']}",
        f"  Manifestation: {ex['failure_manifestation']}",
        f"  Turns       : {ex['executor_turn_count']}",
        f"  Obs flags   : {flag_str}",
        f"  Correctness : {scores['correctness']}  "
          f"Completeness: {scores['completeness']}  "
          f"Efficiency: {scores['efficiency']}  "
          f"Pass: {scores['pass_fail']}",
        f"  Trace:",
    ]
    for line in ex["trace_summary"].split("\n"):
        lines.append(f"    {line}")
    return "\n".join(lines)


def audit_pair(cls_a: str, cls_b: str, label: str) -> tuple[str, list[str]]:
    """Returns (console_output, list_of_findings)."""
    samples_a = load_samples(cls_a, N_SAMPLES, SEED)
    samples_b = load_samples(cls_b, N_SAMPLES, SEED)

    sep  = "-" * 65
    sep2 = "=" * 65

    lines = [
        sep2,
        f"{label}: {cls_a.upper()}  vs  {cls_b.upper()}",
        f"Question: Can you tell these apart from the trace alone?",
        sep2,
    ]

    for i, ex in enumerate(samples_a):
        lines.append(f"\n[{cls_a}]")
        lines.append(fmt_example(ex, i))
        lines.append("")

    lines.append(sep)

    for i, ex in enumerate(samples_b):
        lines.append(f"\n[{cls_b}]")
        lines.append(fmt_example(ex, i))
        lines.append("")

    lines.append(sep2)

    # Auto-analysis: compute distinguishing features
    findings = _auto_analysis(cls_a, samples_a, cls_b, samples_b)
    lines.append("AUTO-ANALYSIS")
    lines.append(sep)
    lines += findings
    lines.append("")

    return "\n".join(lines), findings


def _auto_analysis(
    cls_a: str, samples_a: list[dict],
    cls_b: str, samples_b: list[dict],
) -> list[str]:
    def stats(samples):
        flags_all  = [json.loads(e["observer_flags"]) for e in samples]
        scores_all = [json.loads(e["critic_scores"])   for e in samples]
        flag_counts = [len(f) for f in flags_all]
        flag_types  = set(
            f["flag_type"] for flags in flags_all for f in flags
        )
        avg_turns = sum(e["executor_turn_count"] for e in samples) / len(samples)
        avg_corr  = sum(s["correctness"] for s in scores_all) / len(scores_all)
        avg_eff   = sum(s["efficiency"]  for s in scores_all) / len(scores_all)
        llm_turns = sum(
            e["trace_summary"].count("[LLM]") for e in samples
        ) / len(samples)
        error_turns = sum(
            e["trace_summary"].count("Error:") + e["trace_summary"].count("ValidationError")
            for e in samples
        ) / len(samples)
        return {
            "avg_flags":   sum(flag_counts) / len(flag_counts),
            "flag_types":  flag_types,
            "avg_turns":   avg_turns,
            "avg_corr":    avg_corr,
            "avg_eff":     avg_eff,
            "avg_llm":     llm_turns,
            "avg_errors":  error_turns,
        }

    sa = stats(samples_a)
    sb = stats(samples_b)

    findings = []

    # Flag signal
    if sa["avg_flags"] != sb["avg_flags"]:
        if sa["avg_flags"] > sb["avg_flags"]:
            findings.append(
                f"  DISTINGUISHABLE via flags: {cls_a} fires {sa['avg_flags']:.1f} flags/example "
                f"vs {cls_b} fires {sb['avg_flags']:.1f}."
            )
        else:
            findings.append(
                f"  DISTINGUISHABLE via flags: {cls_b} fires {sb['avg_flags']:.1f} flags/example "
                f"vs {cls_a} fires {sa['avg_flags']:.1f}."
            )
    else:
        findings.append(
            f"  WEAK via flags: both classes fire similar flag counts "
            f"({sa['avg_flags']:.1f} vs {sb['avg_flags']:.1f})."
        )

    # Turn count
    turn_diff = abs(sa["avg_turns"] - sb["avg_turns"])
    if turn_diff >= 2.0:
        hi, hi_v, lo, lo_v = (cls_a, sa["avg_turns"], cls_b, sb["avg_turns"]) if sa["avg_turns"] > sb["avg_turns"] \
            else (cls_b, sb["avg_turns"], cls_a, sa["avg_turns"])
        findings.append(
            f"  DISTINGUISHABLE via turns: {hi} uses {hi_v:.1f} turns avg, "
            f"{lo} uses {lo_v:.1f} turns avg (diff={turn_diff:.1f})."
        )
    else:
        findings.append(
            f"  WEAK via turns: similar turn counts "
            f"({cls_a}={sa['avg_turns']:.1f} vs {cls_b}={sb['avg_turns']:.1f}, diff={turn_diff:.1f})."
        )

    # LLM turn ratio
    llm_diff = abs(sa["avg_llm"] - sb["avg_llm"])
    if llm_diff >= 1.5:
        hi, hi_v, lo, lo_v = (cls_a, sa["avg_llm"], cls_b, sb["avg_llm"]) \
            if sa["avg_llm"] > sb["avg_llm"] else (cls_b, sb["avg_llm"], cls_a, sa["avg_llm"])
        findings.append(
            f"  DISTINGUISHABLE via LLM turns: {hi} has {hi_v:.1f} [LLM] turns avg "
            f"vs {lo} has {lo_v:.1f} (diff={llm_diff:.1f})."
        )
    else:
        findings.append(
            f"  WEAK via LLM turns: similar [LLM] turn counts "
            f"({cls_a}={sa['avg_llm']:.1f} vs {cls_b}={sb['avg_llm']:.1f})."
        )

    # Error presence
    err_diff = abs(sa["avg_errors"] - sb["avg_errors"])
    if err_diff >= 1.0:
        hi, hi_v, lo, lo_v = (cls_a, sa["avg_errors"], cls_b, sb["avg_errors"]) \
            if sa["avg_errors"] > sb["avg_errors"] else (cls_b, sb["avg_errors"], cls_a, sa["avg_errors"])
        findings.append(
            f"  DISTINGUISHABLE via errors: {hi} shows {hi_v:.1f} Error lines avg "
            f"vs {lo} shows {lo_v:.1f}."
        )

    # Correctness
    corr_diff = abs(sa["avg_corr"] - sb["avg_corr"])
    if corr_diff >= 0.3:
        findings.append(
            f"  WEAK SIGNAL via correctness: {cls_a}={sa['avg_corr']:.2f} "
            f"vs {cls_b}={sb['avg_corr']:.2f} (diff={corr_diff:.2f})."
        )

    # Overall verdict
    strong = sum(1 for f in findings if "DISTINGUISHABLE" in f)
    weak   = sum(1 for f in findings if "WEAK" in f)
    if strong >= 2:
        verdict = "VERDICT: Boundary looks CLEAN -- classes are distinguishable on multiple signals."
    elif strong == 1:
        verdict = "VERDICT: Boundary is BORDERLINE -- distinguishable on 1 signal only. Monitor closely."
    else:
        verdict = "VERDICT: Boundary is BLURRY -- classes share most signals. High confusion risk."

    findings.append("")
    findings.append(f"  {verdict}")
    return findings


def main():
    random.seed(SEED)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    all_output = []
    all_output.append("ARIA Confusion Audit -- Research Cycle 0.5")
    all_output.append("=" * 65)
    all_output.append(f"Samples per class: {N_SAMPLES}   Seed: {SEED}")
    all_output.append("Hard pairs tested: prompt_drift/goal_misalignment, "
                      "context_overflow/tool_misuse, hallucination_loop/goal_misalignment")
    all_output.append("")

    verdicts = []

    for cls_a, cls_b, label in HARD_PAIRS:
        output, findings = audit_pair(cls_a, cls_b, label)
        print(output)
        all_output.append(output)

        verdict_line = next((f for f in findings if "VERDICT" in f), "")
        verdicts.append(f"  {label} ({cls_a} vs {cls_b}): {verdict_line.strip()}")

    # Summary
    summary = [
        "",
        "=" * 65,
        "SUMMARY",
        "=" * 65,
        "",
    ] + verdicts + [
        "",
        "Next step: if all verdicts are CLEAN or BORDERLINE -> proceed to DSPy compilation.",
        "If any verdict is BLURRY -> fix data boundary first, then re-run audit.",
        "=" * 65,
    ]

    print("\n".join(summary))
    all_output += summary

    report_path = REPORT_DIR / "confusion_audit_v1.md"
    report_path.write_text("\n".join(all_output), encoding="utf-8")
    print(f"\nFull report saved -> {report_path}")


if __name__ == "__main__":
    main()
