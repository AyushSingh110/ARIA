#!/usr/bin/env python3
"""Compute ARIA vs GAIA ground truth agreement.

Reads all results from data/gaia/results/
Computes:
  - Answer accuracy (ARIA agent got the right GAIA answer)
  - ARIA failure distribution
  - requirement_satisfaction distribution
  - Per-level breakdown

Run:
  cd backend
  python scripts/gaia_agreement.py
  python scripts/gaia_agreement.py --save   # saves report to data/gaia/agreement_report.json
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import click

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

GAIA_DIR     = Path("data/gaia")
RESULTS_DIR  = GAIA_DIR / "results"
REPORT_FILE  = GAIA_DIR / "agreement_report.json"


def load_results() -> list[dict]:
    if not RESULTS_DIR.exists():
        print(f"ERROR: {RESULTS_DIR} not found. Run gaia_run_batch.py first.")
        sys.exit(1)
    results = []
    for f in sorted(RESULTS_DIR.glob("*.json")):
        try:
            r = json.loads(f.read_text(encoding="utf-8"))
            if not r.get("run_error"):
                results.append(r)
        except Exception:
            pass
    return results


def compute_report(results: list[dict]) -> dict:
    if not results:
        return {"error": "No results found."}

    total = len(results)

    # Answer accuracy (auto-check via fuzzy string match)
    answered   = [r for r in results if r.get("gaia_correct") is not None]
    correct    = [r for r in answered if r.get("gaia_correct")]
    acc_pct    = round(len(correct) / len(answered) * 100, 1) if answered else 0.0

    # ARIA failure distribution
    aria_dist  = Counter(r.get("aria_label") or "none" for r in results)
    aria_pct   = {k: round(v / total * 100, 1) for k, v in aria_dist.items()}

    # Requirement satisfaction
    req_sats   = [r.get("requirement_satisfaction", 0.0) for r in results]
    avg_req_sat = round(sum(req_sats) / len(req_sats), 3)

    # Pass rate (req_sat >= 0.75)
    pass_count  = sum(1 for r in req_sats if r >= 0.75)
    pass_rate   = round(pass_count / total * 100, 1)

    # Breakdown by GAIA level
    by_level = {}
    for lvl in sorted({r.get("level", 1) for r in results}):
        lvl_results = [r for r in results if r.get("level") == lvl]
        lvl_correct = [r for r in lvl_results if r.get("gaia_correct")]
        lvl_answered = [r for r in lvl_results if r.get("gaia_correct") is not None]
        by_level[str(lvl)] = {
            "total":    len(lvl_results),
            "answered": len(lvl_answered),
            "correct":  len(lvl_correct),
            "acc_pct":  round(len(lvl_correct) / len(lvl_answered) * 100, 1) if lvl_answered else 0.0,
            "aria_dist": dict(Counter(r.get("aria_label") or "none" for r in lvl_results)),
            "avg_req_sat": round(
                sum(r.get("requirement_satisfaction", 0.0) for r in lvl_results) / len(lvl_results), 3
            ),
        }

    # Correlation: when ARIA says "none" (clean run), how often is gaia_correct True?
    clean_runs = [r for r in answered if not r.get("aria_label")]
    clean_correct = sum(1 for r in clean_runs if r.get("gaia_correct"))
    clean_acc = round(clean_correct / len(clean_runs) * 100, 1) if clean_runs else 0.0

    # Correlation: when ARIA detects a failure, how often is gaia_correct False?
    failure_runs = [r for r in answered if r.get("aria_label")]
    failure_wrong = sum(1 for r in failure_runs if not r.get("gaia_correct"))
    failure_detect_acc = round(failure_wrong / len(failure_runs) * 100, 1) if failure_runs else 0.0

    # Human-labeled agreement (if any labeled results exist)
    human_labeled = [r for r in results if r.get("reviewed") and r.get("human_label") is not None]
    human_agreement = None
    if human_labeled:
        matching = sum(
            1 for r in human_labeled
            if (r.get("aria_label") or "none") == r["human_label"]
        )
        human_agreement = round(matching / len(human_labeled) * 100, 1)

    return {
        "total_runs":            total,
        "answer_accuracy_pct":   acc_pct,
        "answered_count":        len(answered),
        "correct_count":         len(correct),
        "avg_requirement_satisfaction": avg_req_sat,
        "pass_rate_pct":         pass_rate,
        "aria_failure_distribution": dict(aria_dist),
        "aria_failure_pct":      aria_pct,
        "by_level":              by_level,
        "clean_run_accuracy_pct":  clean_acc,
        "failure_detection_acc_pct": failure_detect_acc,
        "human_labeled_runs":    len(human_labeled),
        "human_agreement_pct":   human_agreement,
    }


def print_report(report: dict) -> None:
    print("\n" + "=" * 55)
    print("  ARIA-GAIA Agreement Report")
    print("=" * 55)
    print(f"  Total runs          : {report['total_runs']}")
    print(f"  Answer accuracy     : {report['answer_accuracy_pct']}%  "
          f"({report['correct_count']}/{report['answered_count']} answered)")
    print(f"  Avg req satisfaction: {report['avg_requirement_satisfaction']:.3f}")
    print(f"  Pass rate (≥0.75)   : {report['pass_rate_pct']}%")
    print()
    print("  ARIA Failure Distribution:")
    for cls, pct in sorted(report["aria_failure_pct"].items(), key=lambda x: -x[1]):
        count = report["aria_failure_distribution"].get(cls, 0)
        print(f"    {cls:<22} {pct:>5.1f}%  ({count})")
    print()
    print("  Diagnostic correlation:")
    print(f"    When ARIA says 'clean' → agent was correct: {report['clean_run_accuracy_pct']}%")
    print(f"    When ARIA flags failure → agent was wrong:  {report['failure_detection_acc_pct']}%")
    print()
    print("  By GAIA Level:")
    for lvl, data in report["by_level"].items():
        print(f"    Level {lvl}: {data['total']} tasks | acc={data['acc_pct']}% | "
              f"req_sat={data['avg_req_sat']:.2f}")
    if report.get("human_agreement_pct") is not None:
        print(f"\n  Human-labeled agreement : {report['human_agreement_pct']}%  "
              f"({report['human_labeled_runs']} labeled)")
    print("=" * 55)


@click.command()
@click.option("--save", is_flag=True, help="Save report to data/gaia/agreement_report.json")
def main(save: bool):
    """Compute ARIA vs GAIA ground truth agreement metrics."""
    results = load_results()
    if not results:
        print("No completed results found. Run gaia_run_batch.py first.")
        return

    report = compute_report(results)
    print_report(report)

    if save:
        REPORT_FILE.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nReport saved to {REPORT_FILE}")


if __name__ == "__main__":
    main()
