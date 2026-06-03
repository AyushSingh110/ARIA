#!/usr/bin/env python3
"""Research Cycle 1.25 — Deep audit of gap-labeled runs.

For each gap run, collects:
  - Missing requirement (what the agent failed to satisfy)
  - Failure mechanism (HOW it failed)
  - Failure outcome (WHAT the user experienced)
  - Cluster assignment (A=partial_completion, B=requirement_omission, C=superficial_success)

Produces: research/gap_failure_analysis_v1.md

Run: python scripts/audit_gaps.py
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

RESULTS_DIR = Path("data/realbench/results")
REPORT_DIR  = Path("../research")

CLUSTERS = {
    "A": "partial_completion    -- agent did PART of the task, stopped early",
    "B": "requirement_omission  -- agent finished but IGNORED a specific constraint",
    "C": "superficial_success   -- agent appeared done but used wrong/no source",
    "D": "other                 -- does not fit A/B/C",
}


def load_gap_runs() -> list[dict]:
    files = sorted(RESULTS_DIR.glob("rb_*.json"))
    gaps  = []
    for f in files:
        r = json.loads(f.read_text(encoding="utf-8"))
        if r.get("human_label") == "gap" and not r.get("run_error"):
            gaps.append(r)
    return gaps


def display_run(r: dict) -> None:
    sep  = "-" * 65
    sep2 = "=" * 65
    scores = r.get("critic_scores") or {}
    flags  = r.get("observer_flags") or []
    flag_str = ", ".join(f["flag_type"] for f in flags) if flags else "none"

    print(f"\n{sep2}")
    print(f"  Task ID   : {r['task_id']}")
    print(f"  Task      : {r['task']}")
    print(f"  ARIA said : {r.get('aria_label') or 'none'} (conf={r.get('aria_confidence', 0):.2f})")
    print(f"  You said  : gap")
    print(sep)
    print(f"  Turns     : {r.get('executor_turn_count', 0)}")
    print(f"  Obs flags : {flag_str}")
    print(f"  Critic    : corr={scores.get('correctness','?')}  "
          f"comp={scores.get('completeness','?')}  "
          f"eff={scores.get('efficiency','?')}  "
          f"pass={scores.get('pass_fail','?')}")
    print(sep)
    print("  Trace:")
    for line in r.get("trace_summary", "").split("\n"):
        print(f"    {line[:110]}")
    print(sep)
    print("  Output:")
    print(f"    {r.get('executor_output', '')[:300]}")
    print(sep)
    print("  ARIA reasoning:")
    print(f"    {r.get('aria_reasoning', '')[:200]}")
    print(sep2)


def ask_cluster() -> str:
    print("\n  Cluster options:")
    for k, v in CLUSTERS.items():
        print(f"    {k}. {v}")
    while True:
        val = input("  Cluster (A/B/C/D): ").strip().upper()
        if val in CLUSTERS:
            return val
        print("  Enter A, B, C, or D.")


def ask_text(prompt: str) -> str:
    val = input(f"  {prompt}: ").strip()
    return val if val else "(not specified)"


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    gaps = load_gap_runs()

    if not gaps:
        print("No gap-labeled runs found. Run review_realbench.py first.")
        return

    print(f"\nGap Audit — {len(gaps)} runs labeled 'gap'")
    print("For each run: read the trace, then answer 3 questions.\n")

    audit_records = []

    for i, r in enumerate(gaps):
        display_run(r)

        print(f"\n  [{i+1}/{len(gaps)}] Answer these 3 questions:")
        missing_req = ask_text("What requirement did the agent miss or ignore?")
        mechanism   = ask_text("HOW did it fail? (e.g. 'searched but ignored quantity', 'answered from memory')")
        cluster     = ask_cluster()

        record = {
            "task_id":           r["task_id"],
            "task":              r["task"],
            "aria_label":        r.get("aria_label") or "none",
            "missing_requirement": missing_req,
            "failure_mechanism": mechanism,
            "cluster":           cluster,
            "cluster_name":      CLUSTERS[cluster].split("--")[0].strip(),
            "critic_correctness": (r.get("critic_scores") or {}).get("correctness", 0),
            "critic_pass":        (r.get("critic_scores") or {}).get("pass_fail", False),
            "turns":             r.get("executor_turn_count", 0),
        }
        audit_records.append(record)

        result_path = RESULTS_DIR / f"{r['task_id']}.json"
        full = json.loads(result_path.read_text(encoding="utf-8"))
        full["gap_audit"] = record
        result_path.write_text(json.dumps(full, indent=2), encoding="utf-8")
        print(f"  Saved audit for {r['task_id']}.")

    # ── Generate report ───────────────────────────────────────────
    cluster_counts = Counter(rec["cluster"] for rec in audit_records)
    SEP  = "-" * 65
    SEP2 = "=" * 65

    lines = [
        "ARIA Gap Failure Analysis -- Research Cycle 1.25",
        SEP2,
        f"Total gap runs audited: {len(audit_records)}",
        "",
        "CLUSTER DISTRIBUTION",
        SEP,
    ]

    for k, label in CLUSTERS.items():
        count = cluster_counts.get(k, 0)
        pct   = count / len(audit_records) * 100 if audit_records else 0
        lines.append(f"  {k}. {label}")
        lines.append(f"     Count: {count} ({pct:.0f}%)")

    lines += ["", "INDIVIDUAL FINDINGS", SEP]

    for rec in audit_records:
        lines += [
            f"",
            f"  [{rec['task_id']}] Cluster {rec['cluster']} -- {rec['cluster_name']}",
            f"  Task      : {rec['task'][:80]}",
            f"  ARIA said : {rec['aria_label']}",
            f"  Missing   : {rec['missing_requirement']}",
            f"  Mechanism : {rec['failure_mechanism']}",
            f"  Critic    : correctness={rec['critic_correctness']}  pass={rec['critic_pass']}",
        ]

    # Candidate taxonomy additions
    lines += [
        "",
        SEP2,
        "CANDIDATE TAXONOMY ADDITIONS FOR v2",
        SEP,
        "",
        "Based on the cluster analysis, the following sub-types of",
        "goal_misalignment are proposed for explicit taxonomy coverage:",
        "",
    ]

    a_count = cluster_counts.get("A", 0)
    b_count = cluster_counts.get("B", 0)
    c_count = cluster_counts.get("C", 0)

    if a_count:
        lines += [
            f"  partial_completion ({a_count} cases)",
            "    Agent completes a subset of a multi-step task and terminates.",
            "    Signal: executor_output addresses step 1 only; task requires N steps.",
            "",
        ]
    if b_count:
        lines += [
            f"  requirement_omission ({b_count} cases)",
            "    Agent completes task but ignores an explicit constraint in the prompt.",
            "    Signal: output is present but violates a stated requirement.",
            "",
        ]
    if c_count:
        lines += [
            f"  superficial_success ({c_count} cases)",
            "    Agent produces a confident complete-looking answer without the",
            "    required evidence (tool call, source citation, real-time data).",
            "    Signal: high confidence, low turns, no tool calls for factual claims.",
            "",
        ]

    lines += [
        SEP2,
        "",
        "CRITIC CALIBRATION INPUT",
        SEP,
        "",
        "Correctness scores of gap runs (input for threshold recalibration):",
    ]

    scores = [rec["critic_correctness"] for rec in audit_records if rec["critic_correctness"]]
    if scores:
        avg = sum(scores) / len(scores)
        mn  = min(scores)
        mx  = max(scores)
        passed = sum(1 for rec in audit_records if rec["critic_pass"])
        lines += [
            f"  Min correctness : {mn}",
            f"  Max correctness : {mx}",
            f"  Avg correctness : {avg:.2f}",
            f"  Critic passed   : {passed}/{len(audit_records)} gap runs",
            "",
            "  If the Critic is passing these runs, the pass threshold (3.5)",
            "  is too high or the correctness scoring is not penalising",
            "  requirement omissions sufficiently.",
        ]

    lines.append("")

    full_report = "\n".join(lines)
    print("\n" + full_report.encode("ascii", errors="replace").decode("ascii"))

    report_path = REPORT_DIR / "gap_failure_analysis_v1.md"
    report_path.write_text(full_report, encoding="utf-8")
    print(f"\nReport saved -> {report_path}")


if __name__ == "__main__":
    main()
