#!/usr/bin/env python3
"""Freeze a stratified test set ONCE and lock it (Step 1.5).

This is the single most important methodological act in the project: it splits
the gold-labeled traces into a frozen test set you never tune against, and a
train/dev pool you develop on. Reporting a number on a test set that was locked
*before* you touched the system is what makes it defensible.

  python scripts/freeze_test_set.py --gold data/labels/gold.jsonl --test-size 70

Outputs (under data/splits/):
  test.jsonl     — frozen test traces (real, double-labeled). DO NOT OPEN while developing.
  train.jsonl    — everything else (combine with injected synthetic for training).
  test_ids.json  — just the test trace_ids, for the train-exclusion guard.
  manifest.json  — seed, per-class counts, and a sha256 of the test id list.

Safety: if a manifest already exists the tool REFUSES to overwrite unless you
pass --force (which prints a loud warning). Re-freezing after you've seen test
results silently destroys the guarantee — that's why it's gated.

Stratified = each class keeps (approximately) its overall proportion in the test
split, so a 70-trace test set mirrors the class balance of the labeled pool.
"""
from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import click

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aria.eval.dataset import trace_by_id  # noqa: E402

SPLITS_DIR = Path("data/splits")
TEST_FP = SPLITS_DIR / "test.jsonl"
TRAIN_FP = SPLITS_DIR / "train.jsonl"
TEST_IDS_FP = SPLITS_DIR / "test_ids.json"
MANIFEST_FP = SPLITS_DIR / "manifest.json"


def _load_gold(path: Path) -> dict[str, str]:
    if not path.exists():
        print(f"ERROR: {path} not found. Build it via compute_kappa.py --adjudicate.")
        sys.exit(1)
    gold: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        gold[rec["trace_id"]] = rec["label"]
    return gold


def _stratified_test_ids(gold: dict[str, str], test_size: int, seed: int) -> list[str]:
    """Pick test_size ids keeping each class's proportion (>=1 per present class)."""
    import random
    rng = random.Random(seed)

    by_class: dict[str, list[str]] = defaultdict(list)
    for tid, lbl in gold.items():
        by_class[lbl].append(tid)
    for ids in by_class.values():
        rng.shuffle(ids)

    total = len(gold)
    test_size = min(test_size, total)
    chosen: list[str] = []
    # proportional quota per class, at least 1 where the class exists
    for lbl, ids in by_class.items():
        quota = max(1, round(test_size * len(ids) / total))
        chosen.extend(ids[:quota])
    # trim/pad to exactly test_size deterministically
    rng.shuffle(chosen)
    if len(chosen) > test_size:
        chosen = chosen[:test_size]
    elif len(chosen) < test_size:
        remaining = [t for t in gold if t not in set(chosen)]
        rng.shuffle(remaining)
        chosen.extend(remaining[: test_size - len(chosen)])
    return chosen


@click.command()
@click.option("--gold", "gold_path", default="data/labels/gold.jsonl", show_default=True)
@click.option("--test-size", default=70, show_default=True, help="Frozen test-set size")
@click.option("--seed", default=20260613, show_default=True)
@click.option("--force", is_flag=True, help="Overwrite an existing frozen split (DANGEROUS)")
def main(gold_path, test_size, seed, force):
    """Create and lock the stratified test set."""
    if MANIFEST_FP.exists() and not force:
        m = json.loads(MANIFEST_FP.read_text(encoding="utf-8"))
        print(f"REFUSING to re-freeze. A locked split already exists "
              f"(created {m.get('created')}, test n={m.get('test_size')}).")
        print("Re-freezing after seeing test results destroys the guarantee.")
        print("If you are SURE, pass --force.")
        sys.exit(1)

    gold = _load_gold(Path(gold_path))
    by_id = trace_by_id()
    missing = [t for t in gold if t not in by_id]
    if missing:
        print(f"WARNING: {len(missing)} gold trace_ids have no matching result file "
              f"(skipped). e.g. {missing[:3]}")
    gold = {t: l for t, l in gold.items() if t in by_id}
    if not gold:
        print("ERROR: no gold labels map to existing traces.")
        sys.exit(1)

    test_ids = set(_stratified_test_ids(gold, test_size, seed))
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)

    def _rec(tid: str) -> dict:
        return {"trace_id": tid, "gold_label": gold[tid], "benchmark": by_id[tid].benchmark}

    test_recs = [_rec(t) for t in gold if t in test_ids]
    train_recs = [_rec(t) for t in gold if t not in test_ids]

    TEST_FP.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in test_recs) + "\n", encoding="utf-8")
    TRAIN_FP.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in train_recs) + "\n", encoding="utf-8")

    sorted_ids = sorted(test_ids)
    TEST_IDS_FP.write_text(json.dumps(sorted_ids, indent=2), encoding="utf-8")
    test_hash = hashlib.sha256("\n".join(sorted_ids).encode()).hexdigest()

    manifest = {
        "created": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "gold_source": gold_path,
        "total_labeled": len(gold),
        "test_size": len(test_recs),
        "train_size": len(train_recs),
        "test_class_balance": dict(Counter(r["gold_label"] for r in test_recs)),
        "train_class_balance": dict(Counter(r["gold_label"] for r in train_recs)),
        "test_ids_sha256": test_hash,
    }
    MANIFEST_FP.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"\nFROZEN TEST SET LOCKED  (sha256 {test_hash[:12]}…)")
    print(f"  test  : {len(test_recs)}  -> {TEST_FP}")
    print(f"  train : {len(train_recs)} -> {TRAIN_FP}")
    print(f"  test class balance: {manifest['test_class_balance']}")
    print("\nDo NOT tune rules/thresholds/prompts against test.jsonl.")
    print("Use aria.eval.dataset + data/splits/test_ids.json to exclude test from training.")


if __name__ == "__main__":
    main()
