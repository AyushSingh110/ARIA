#!/usr/bin/env python3
"""Print data quality statistics for all synthetic JSONL files.

Run: python scripts/data_stats.py
"""
import json
from collections import Counter
from pathlib import Path

DATA_DIR = Path("data/synthetic")

CLASSES = [
    "none",
    "prompt_drift",
    "tool_misuse",
    "context_overflow",
    "goal_misalignment",
    "hallucination_loop",
]


def main():
    print("ARIA-Bench — Data Quality Report")
    print("=" * 65)

    total = 0
    for cls in CLASSES:
        path = DATA_DIR / f"{cls}.jsonl"
        if not path.exists():
            print(f"  {cls}: FILE MISSING")
            continue

        examples = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
        total += len(examples)

        manifests = Counter(e["failure_manifestation"] for e in examples)
        flag_types = Counter()
        for e in examples:
            for f in json.loads(e["observer_flags"]):
                flag_types[f["flag_type"]] += 1

        scores_list = [json.loads(e["critic_scores"]) for e in examples]
        pass_count = sum(1 for s in scores_list if s["pass_fail"])
        avg_corr = sum(s["correctness"] for s in scores_list) / len(scores_list)
        avg_turns = sum(e["executor_turn_count"] for e in examples) / len(examples)
        has_drift = sum(1 for e in examples if e.get("drift_scores"))

        print(f"\n[{cls}]  ({len(examples)} examples)")
        print(f"  Manifestation subtypes : {dict(manifests)}")
        print(f"  Observer flags fired   : {dict(flag_types) if flag_types else 'none'}")
        print(f"  pass_fail = True       : {pass_count}/{len(examples)}")
        print(f"  Avg critic correctness : {avg_corr:.2f}")
        print(f"  Avg executor turns     : {avg_turns:.1f}")
        print(f"  Has drift_scores field : {has_drift}/{len(examples)}")

    print("\n" + "=" * 65)
    print(f"Total examples: {total}  ({total // len(CLASSES)} per class)")
    print("=" * 65)


if __name__ == "__main__":
    main()
