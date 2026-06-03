#!/usr/bin/env python3
"""Compare two validation summaries: v1 (before data fix) vs v2 (after).

Run: python scripts/compare_validation_runs.py

Reads:
  data/compiled/val_summary_v1.json  (old run)
  data/compiled/val_summary.json     (new run)
"""
from __future__ import annotations
import json
from pathlib import Path

OLD_PATH = Path("data/compiled/val_summary_v1.json")
NEW_PATH = Path("data/compiled/val_summary.json")

CLASSES = [
    "prompt_drift", "tool_misuse", "context_overflow",
    "goal_misalignment", "hallucination_loop", "none",
]


def load(path: Path) -> dict:
    if not path.exists():
        print(f"ERROR: {path} not found.")
        raise SystemExit(1)
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    old = load(OLD_PATH)
    new = load(NEW_PATH)

    SEP  = "-" * 68
    SEP2 = "=" * 68

    print(f"\n{SEP2}")
    print(f"  ARIA DSPy -- Validation Comparison: v1 vs v2")
    print(f"  v1 = before data fix  |  v2 = after data fix")
    print(SEP2)

    # Overall accuracy
    old_acc = old.get("accuracy", 0)
    new_acc = new.get("accuracy", 0)
    delta   = new_acc - old_acc
    sign    = "+" if delta >= 0 else ""
    print(f"\n  Overall accuracy:  v1={old_acc:.1%}  v2={new_acc:.1%}  delta={sign}{delta:.1%}")

    # Per-class F1 comparison
    print(f"\n  {'Class':<22}  {'v1 F1':>6}  {'v2 F1':>6}  {'Delta':>7}  {'v1 Rec':>7}  {'v2 Rec':>7}")
    print(f"  {SEP}")

    for cls in CLASSES:
        old_c = old.get("per_class", {}).get(cls, {})
        new_c = new.get("per_class", {}).get(cls, {})
        old_f1  = old_c.get("f1", 0.0)
        new_f1  = new_c.get("f1", 0.0)
        old_rec = old_c.get("recall", 0.0)
        new_rec = new_c.get("recall", 0.0)
        df1 = new_f1 - old_f1
        sign = "+" if df1 >= 0 else ""
        flag = " <-- IMPROVED" if df1 > 0.05 else (" <-- REGRESSED" if df1 < -0.05 else "")
        print(f"  {cls:<22}  {old_f1:>5.1%}  {new_f1:>5.1%}  {sign}{df1:>6.1%}  "
              f"{old_rec:>6.1%}  {new_rec:>6.1%}{flag}")

    # Hard pair comparison
    old_hp = old.get("hard_pair", {})
    new_hp = new.get("hard_pair", {})
    print(f"\n  HARD PAIR: hallucination_loop <-> goal_misalignment")
    print(f"  {SEP}")
    print(f"  hl->gm confusions:  v1={old_hp.get('hl_as_gm',0)}  v2={new_hp.get('hl_as_gm',0)}")
    print(f"  gm->hl confusions:  v1={old_hp.get('gm_as_hl',0)}  v2={new_hp.get('gm_as_hl',0)}")

    # Confusion matrix delta for the two problem classes
    print(f"\n  CONFUSION MATRIX DELTA (goal_misalignment row)")
    print(f"  {SEP}")
    old_cm = old.get("confusion_matrix", {}).get("goal_misalignment", {})
    new_cm = new.get("confusion_matrix", {}).get("goal_misalignment", {})
    for pred_cls in CLASSES:
        old_n = old_cm.get(pred_cls, 0)
        new_n = new_cm.get(pred_cls, 0)
        if old_n != new_n or pred_cls == "goal_misalignment":
            flag = " <-- FIXED" if (pred_cls == "tool_misuse" and new_n < old_n) else ""
            print(f"  gold=goal_misalignment predicted={pred_cls:<22}  "
                  f"v1={old_n}  v2={new_n}{flag}")

    print(f"\n  CONFUSION MATRIX DELTA (hallucination_loop row)")
    print(f"  {SEP}")
    old_cm2 = old.get("confusion_matrix", {}).get("hallucination_loop", {})
    new_cm2 = new.get("confusion_matrix", {}).get("hallucination_loop", {})
    for pred_cls in CLASSES:
        old_n = old_cm2.get(pred_cls, 0)
        new_n = new_cm2.get(pred_cls, 0)
        if old_n != new_n or pred_cls == "hallucination_loop":
            flag = " <-- FIXED" if (pred_cls == "goal_misalignment" and new_n < old_n) else ""
            print(f"  gold=hallucination_loop predicted={pred_cls:<22}  "
                  f"v1={old_n}  v2={new_n}{flag}")

    print(f"\n{SEP2}\n")


if __name__ == "__main__":
    main()
