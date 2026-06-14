#!/usr/bin/env python3
"""Report defensible metrics on the frozen test set (Step 2.3).

Computes, against gold labels:
  * overall accuracy with a bootstrap 95% CI (small-N honesty)
  * macro-F1 with a bootstrap 95% CI
  * per-class precision / recall / F1 / support
  * confusion matrix (gold rows × pred cols)
  * Cohen's kappa between ARIA and gold

Predictions default to ARIA's stored full-system label (aria_label) joined to the
frozen test traces. Pass --pred-file to score an ablation's predictions instead.

  # full system on the frozen test set:
  python scripts/eval_metrics.py
  # a specific ablation config's predictions:
  python scripts/eval_metrics.py --pred-file data/ablation/a.jsonl --tag A
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import click

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aria.eval.dataset import trace_by_id  # noqa: E402
from aria.eval import metrics as M  # noqa: E402

TEST_FP = Path("data/splits/test.jsonl")


def _load_gold(split_fp: Path) -> dict[str, str]:
    if not split_fp.exists():
        print(f"ERROR: {split_fp} not found. Freeze a test set first "
              f"(scripts/freeze_test_set.py).")
        sys.exit(1)
    gold = {}
    for line in split_fp.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            gold[r["trace_id"]] = r["gold_label"]
    return gold


def _load_pred_file(fp: Path) -> dict[str, str]:
    preds = {}
    for line in fp.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            preds[r["trace_id"]] = (r.get("pred") or r.get("aria_label") or "none")
    return preds


def _report(gold_map: dict[str, str], pred_map: dict[str, str], tag: str) -> dict:
    ids = [t for t in gold_map if t in pred_map]
    if not ids:
        print("No overlap between gold test ids and predictions.")
        sys.exit(1)
    gold = [gold_map[t] for t in ids]
    pred = [pred_map[t] for t in ids]

    acc, acc_lo, acc_hi = M.bootstrap_ci(gold, pred, M.accuracy)
    mf1, mf1_lo, mf1_hi = M.bootstrap_ci(gold, pred, M.macro_f1)
    kappa = M.cohen_kappa(gold, pred)
    per_class = M.per_class_prf(gold, pred)
    labels, conf = M.confusion_matrix(gold, pred)

    print(f"\n{'=' * 64}")
    print(f"  Metrics on frozen test set  [{tag}]   (N={len(ids)})")
    print(f"{'=' * 64}")
    print(f"  Accuracy : {acc:.1%}   95% CI [{acc_lo:.1%}, {acc_hi:.1%}]")
    print(f"  Macro-F1 : {mf1:.3f}   95% CI [{mf1_lo:.3f}, {mf1_hi:.3f}]")
    print(f"  Cohen κ (ARIA vs gold): {kappa:.3f}")
    print(f"\n  Per-class:")
    print(f"    {'class':<20}{'P':>7}{'R':>7}{'F1':>7}{'supp':>6}")
    for c in per_class:
        print(f"    {c.label:<20}{c.precision:>7.2f}{c.recall:>7.2f}{c.f1:>7.2f}{c.support:>6}")
    print(f"\n  Confusion (rows=gold, cols=pred):")
    print("    " + M.format_confusion(labels, conf).replace("\n", "\n    "))
    print(f"{'=' * 64}")

    return {
        "tag": tag, "n": len(ids),
        "accuracy": acc, "accuracy_ci": [acc_lo, acc_hi],
        "macro_f1": mf1, "macro_f1_ci": [mf1_lo, mf1_hi],
        "cohen_kappa_vs_gold": kappa,
        "per_class": [vars(c) for c in per_class],
        "confusion_labels": labels, "confusion_matrix": conf,
    }


@click.command()
@click.option("--split", "split_fp", default=str(TEST_FP), show_default=True,
              help="Frozen gold split (test.jsonl)")
@click.option("--pred-file", default=None,
              help="JSONL of {trace_id, pred}. Default: ARIA's stored aria_label.")
@click.option("--tag", default="full-system", show_default=True)
@click.option("--save", default=None, help="Write the report JSON to this path")
def main(split_fp, pred_file, tag, save):
    """Compute and print test-set metrics."""
    gold_map = _load_gold(Path(split_fp))
    if pred_file:
        pred_map = _load_pred_file(Path(pred_file))
    else:
        # default: full-system prediction stored on each trace
        by_id = trace_by_id()
        pred_map = {t: by_id[t].aria_label_norm for t in gold_map if t in by_id}

    report = _report(gold_map, pred_map, tag)
    if save:
        Path(save).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nSaved report -> {save}")


if __name__ == "__main__":
    main()
