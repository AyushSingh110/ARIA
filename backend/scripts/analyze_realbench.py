#!/usr/bin/env python3
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import click

RESULTS_DIR = Path("data/realbench/results")
REPORT_DIR  = Path("../research")

CLASSES = [
    "none", "prompt_drift", "tool_misuse",
    "context_overflow", "goal_misalignment", "hallucination_loop",
]


def load_results(labeled_only: bool) -> list[dict]:
    if not RESULTS_DIR.exists():
        print("No results found. Run run_realbench.py first.")
        raise SystemExit(1)
    results = []
    for f in sorted(RESULTS_DIR.glob("rb_*.json")):
        r = json.loads(f.read_text(encoding="utf-8"))
        if r.get("run_error"):
            continue
        if labeled_only and not r.get("reviewed"):
            continue
        results.append(r)
    return results


def q1_distribution(results: list[dict]) -> list[str]:
    """What is the real-world distribution of failure classes?"""
    aria_labels   = Counter(r.get("aria_label") or "none" for r in results)
    human_labels  = Counter(r.get("human_label") for r in results if r.get("reviewed"))
    expected      = Counter(r.get("expected_class", "unknown") for r in results)

    SEP  = "-" * 60
    SEP2 = "=" * 60
    lines = [
        SEP2,
        "Q1: REAL-WORLD DISTRIBUTION OF FAILURE CLASSES",
        SEP2,
        "",
        f"  Total runs analysed : {len(results)}",
        f"  Human-reviewed      : {sum(1 for r in results if r.get('reviewed'))}",
        "",
        f"  {'Class':<24}  {'ARIA':>6}  {'Human':>6}  {'Expected':>8}",
        f"  {SEP}",
    ]

    all_classes = sorted(set(list(aria_labels.keys()) + list(human_labels.keys()) + list(expected.keys())))
    for cls in all_classes:
        a = aria_labels.get(cls, 0)
        h = human_labels.get(cls, 0)
        e = expected.get(cls, 0)
        a_pct = a / len(results) * 100 if results else 0
        lines.append(f"  {cls:<24}  {a:>4} ({a_pct:4.0f}%)  {h:>5}  {e:>8}")

    lines += [
        "",
        "  TAXONOMY GAP FLAGS (human labels outside taxonomy):",
    ]
    gap_count   = human_labels.get("gap", 0)
    multi_count = human_labels.get("multi", 0)
    unclear     = human_labels.get("unclear", 0)
    if gap_count:
        lines.append(f"    'gap'    : {gap_count} runs — new failure type not in taxonomy")
    if multi_count:
        lines.append(f"    'multi'  : {multi_count} runs — multiple classes simultaneously")
    if unclear:
        lines.append(f"    'unclear': {unclear} runs — reviewer could not determine")
    if not (gap_count or multi_count or unclear):
        lines.append("    None detected yet.")
    lines.append("")
    return lines


def q2_signal_reliability(results: list[dict]) -> list[str]:
    """Do the observable signals still work on real traces?"""
    SEP  = "-" * 60
    SEP2 = "=" * 60

    # For each ARIA class, check what signals fired
    by_aria_class: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        cls = r.get("aria_label") or "none"
        by_aria_class[cls].append(r)

    lines = [
        SEP2,
        "Q2: OBSERVABLE SIGNAL RELIABILITY ON REAL TRACES",
        SEP2,
        "",
        "  For each diagnosed class: avg turns, flags fired, avg correctness",
        f"  {SEP}",
        f"  {'Class':<24}  {'N':>3}  {'AvgTurns':>9}  {'FlagsAvg':>9}  {'AvgCorr':>8}  {'PassRate':>9}",
        f"  {SEP}",
    ]

    for cls in CLASSES:
        runs = by_aria_class.get(cls, [])
        if not runs:
            continue
        avg_turns = sum(r.get("executor_turn_count", 0) for r in runs) / len(runs)
        avg_flags = sum(len(r.get("observer_flags", [])) for r in runs) / len(runs)
        scores    = [r.get("critic_scores") or {} for r in runs]
        avg_corr  = sum(s.get("correctness", 3.0) for s in scores) / len(scores)
        pass_rate = sum(1 for s in scores if s.get("pass_fail")) / len(scores)
        lines.append(
            f"  {cls:<24}  {len(runs):>3}  {avg_turns:>9.1f}  "
            f"{avg_flags:>9.1f}  {avg_corr:>8.2f}  {pass_rate:>8.0%}"
        )

    lines += [
        "",
        "  SIGNAL VALIDATION (expected patterns from synthetic data):",
        f"  {SEP}",
    ]

    # context_overflow: expect tool_repetition flags
    co_runs = by_aria_class.get("context_overflow", [])
    if co_runs:
        has_rep = sum(1 for r in co_runs
                      if any(f["flag_type"] == "tool_repetition"
                             for f in r.get("observer_flags", [])))
        lines.append(f"  context_overflow: {has_rep}/{len(co_runs)} have tool_repetition flags "
                     f"(expected: most)")

    # tool_misuse: expect tool_error_loop flags
    tm_runs = by_aria_class.get("tool_misuse", [])
    if tm_runs:
        has_err = sum(1 for r in tm_runs
                      if any(f["flag_type"] == "tool_error_loop"
                             for f in r.get("observer_flags", [])))
        lines.append(f"  tool_misuse: {has_err}/{len(tm_runs)} have tool_error_loop flags "
                     f"(expected: most)")

    # goal_misalignment: expect 0 flags
    gm_runs = by_aria_class.get("goal_misalignment", [])
    if gm_runs:
        zero_flags = sum(1 for r in gm_runs if not r.get("observer_flags"))
        lines.append(f"  goal_misalignment: {zero_flags}/{len(gm_runs)} have zero flags "
                     f"(expected: most)")

    # hallucination_loop: expect 0 flags + low correctness
    hl_runs = by_aria_class.get("hallucination_loop", [])
    if hl_runs:
        low_corr = sum(1 for r in hl_runs
                       if (r.get("critic_scores") or {}).get("correctness", 5.0) < 2.5)
        lines.append(f"  hallucination_loop: {low_corr}/{len(hl_runs)} have correctness < 2.5 "
                     f"(expected: most)")

    lines.append("")
    return lines


def q3_taxonomy_gaps(results: list[dict]) -> list[str]:
    """Where does the taxonomy break?"""
    SEP  = "-" * 60
    SEP2 = "=" * 60

    reviewed = [r for r in results if r.get("reviewed")]
    lines = [
        SEP2,
        "Q3: WHERE DOES THE TAXONOMY BREAK?",
        SEP2,
        "",
    ]

    if not reviewed:
        lines += ["  No human-reviewed results yet. Run review_realbench.py.", ""]
        return lines

    # Agreement between ARIA and human.
    # aria_label is None for clean runs but human_label is the string "none" —
    # normalise both so clean-run matches are counted.
    agreed   = sum(
        1 for r in reviewed
        if (r.get("aria_label") or "none") == (r.get("human_label") or "none")
    )
    disagreed = len(reviewed) - agreed
    lines += [
        f"  Human-reviewed runs : {len(reviewed)}",
        f"  ARIA/human agreement: {agreed}/{len(reviewed)} = {agreed/len(reviewed):.0%}",
        "",
        f"  DISAGREEMENTS (ARIA label -> Human label):",
        f"  {SEP}",
    ]

    for r in reviewed:
        a = r.get("aria_label") or "none"
        h = r.get("human_label") or "?"
        if a != h:
            lines.append(f"  {r['task_id']}: ARIA={a:<22} Human={h}")
            if r.get("human_notes"):
                lines.append(f"    Note: {r['human_notes']}")

    # Gap analysis
    gap_runs = [r for r in reviewed if r.get("human_label") == "gap"]
    multi_runs = [r for r in reviewed if r.get("human_label") == "multi"]

    lines += [
        "",
        f"  TAXONOMY GAP ANALYSIS:",
        f"  {SEP}",
    ]

    if gap_runs:
        lines.append(f"  NEW FAILURE TYPES OBSERVED ({len(gap_runs)} cases):")
        for r in gap_runs:
            lines.append(f"    {r['task_id']}: {r['task'][:70]}")
            lines.append(f"      Notes: {r.get('human_notes', 'none')}")
    else:
        lines.append("  No new failure types observed yet.")

    if multi_runs:
        lines.append(f"\n  MULTI-CLASS FAILURES ({len(multi_runs)} cases):")
        for r in multi_runs:
            lines.append(f"    {r['task_id']}: ARIA diagnosed {r.get('aria_label')} but multiple classes present")
            lines.append(f"      Notes: {r.get('human_notes', 'none')}")
    else:
        lines.append("  No multi-class failures observed yet.")

    # Mentor's prediction check
    total = len(results)
    none_count = sum(1 for r in results if (r.get("aria_label") or "none") == "none")
    gm_count   = sum(1 for r in results if (r.get("aria_label") or "none") == "goal_misalignment")
    lines += [
        "",
        f"  MENTOR PREDICTION CHECK:",
        f"  {SEP}",
        f"  Predicted NONE        50-70% | Actual: {none_count}/{total} = {none_count/total:.0%}",
        f"  Predicted GOAL_MISALIGNMENT 15-25% | Actual: {gm_count}/{total} = {gm_count/total:.0%}",
        "",
    ]

    return lines


@click.command()
@click.option("--labeled-only", is_flag=True, help="Only include human-reviewed results")
def main(labeled_only: bool):
    """Analyse ARIA-RealBench results — Research Cycle 1."""
    results = load_results(labeled_only)

    if not results:
        print("No results to analyse.")
        return

    all_lines = [
        "ARIA-RealBench Analysis — Research Cycle 1",
        f"Runs loaded: {len(results)}  |  Labeled only: {labeled_only}",
        "",
    ]

    all_lines += q1_distribution(results)
    all_lines += q2_signal_reliability(results)
    all_lines += q3_taxonomy_gaps(results)

    output = "\n".join(all_lines)
    print(output.encode("ascii", errors="replace").decode("ascii"))

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORT_DIR / "realbench_analysis_v1.md"
    report_path.write_text(output, encoding="utf-8")
    print(f"\nReport saved -> {report_path}")


if __name__ == "__main__":
    main()
