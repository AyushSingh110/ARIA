#!/usr/bin/env python3
"""Two-annotator labeling tool (Step 1.4).

Workflow:
  1. Build ONE shared sample both annotators will label (stratified across
     benchmarks + ARIA's predicted class so rare classes appear):
       python scripts/label_traces.py --make-sample 60 --seed 7

  2. Each annotator labels the SAME sample, independently, without seeing
     ARIA's prediction (hidden to avoid anchoring):
       python scripts/label_traces.py --annotator ayush
       python scripts/label_traces.py --annotator classmate

  3. Compute agreement and adjudicate:
       python scripts/compute_kappa.py --a ayush --b classmate

Labels are written to data/labels/<annotator>.jsonl (one JSON object per line),
keyed by the global trace_id. Labeling is resumable — already-labeled traces are
skipped, so you can stop and resume any time.

Follow docs/labeling-guide.md. The label vocabulary lives in aria.eval.dataset.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import click

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aria.eval.dataset import VALID_LABELS, iter_traces, trace_by_id  # noqa: E402

LABELS_DIR = Path("data/labels")
SAMPLE_FILE = LABELS_DIR / "sample.json"


# ── Sample construction ───────────────────────────────────────────────────────

def _make_sample(n: int, seed: int, benchmarks: list[str] | None) -> list[str]:
    """Pick n trace_ids, round-robin across ARIA-predicted-class buckets.

    We stratify by ARIA's predicted label as a *proxy* for class diversity
    (we have no gold yet). Round-robin guarantees rare predicted classes are
    represented rather than swamped by the majority class.
    """
    import random
    rng = random.Random(seed)

    buckets: dict[str, list[str]] = defaultdict(list)
    for t in iter_traces(benchmarks):
        buckets[t.aria_label_norm].append(t.trace_id)
    for ids in buckets.values():
        rng.shuffle(ids)

    order = sorted(buckets)  # deterministic bucket order
    sample: list[str] = []
    while len(sample) < n and any(buckets[b] for b in order):
        for b in order:
            if buckets[b]:
                sample.append(buckets[b].pop())
                if len(sample) >= n:
                    break
    return sample


# ── Annotation display (ARIA prediction hidden) ──────────────────────────────

def _display(t, idx: int, total: int) -> None:
    sep = "-" * 70
    print(f"\n{'=' * 70}")
    print(f"  [{idx}/{total}]  trace_id: {t.trace_id}   ({t.benchmark})")
    print(sep)
    print(f"  Task: {t.task[:300]}")
    if t.expected_answer:
        print(f"  Known-correct answer: {t.expected_answer[:120]}")
        ac = {True: 'YES', False: 'NO', None: '?'}[t.answer_correct]
        print(f"  Agent got it right?  : {ac}")
    print(sep)
    flags = ", ".join(f["flag_type"] for f in t.observer_flags) or "none"
    cs = t.critic_scores or {}
    print(f"  Observer flags : {flags}")
    print(f"  Req satisfaction: {cs.get('requirement_satisfaction', '?')}")
    print(sep)
    print("  Trace:")
    for line in (t.trace_summary or "(no trace)").split("\n"):
        print(f"    {line[:110]}")
    print(sep)
    print("  Final output:")
    print(f"    {(t.executor_output or '')[:300]}")
    print(f"{'=' * 70}")
    # NOTE: ARIA's own prediction (t.aria_label) is deliberately NOT shown.


def _ask(idx: int, total: int) -> tuple[str, str]:
    menu = "  ".join(f"{i+1}.{l}" for i, l in enumerate(VALID_LABELS))
    print(f"\n  Labels: {menu}")
    print("  s=skip  q=quit")
    while True:
        raw = input("  Label: ").strip().lower()
        if raw in ("q", "quit"):
            return "__quit__", ""
        if raw in ("s", "skip"):
            return "__skip__", ""
        label = None
        if raw.isdigit() and 1 <= int(raw) <= len(VALID_LABELS):
            label = VALID_LABELS[int(raw) - 1]
        elif raw in VALID_LABELS:
            label = raw
        if label:
            notes = input("  Notes (optional): ").strip()
            return label, notes
        print("  Invalid — enter a number, a label name, s, or q.")


# ── Persistence ───────────────────────────────────────────────────────────────

def _labels_path(annotator: str) -> Path:
    return LABELS_DIR / f"{annotator}.jsonl"


def _load_done(annotator: str) -> set[str]:
    fp = _labels_path(annotator)
    if not fp.exists():
        return set()
    done = set()
    for line in fp.read_text(encoding="utf-8").splitlines():
        try:
            done.add(json.loads(line)["trace_id"])
        except Exception:
            pass
    return done


def _append_label(annotator: str, rec: dict) -> None:
    LABELS_DIR.mkdir(parents=True, exist_ok=True)
    with _labels_path(annotator).open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

@click.command()
@click.option("--make-sample", type=int, default=None,
              help="Create the shared labeling sample of this many traces, then exit.")
@click.option("--seed", default=7, show_default=True, help="Sample seed (reproducible).")
@click.option("--benchmarks", default=None,
              help="Comma-separated subset (default: all with results).")
@click.option("--annotator", default=None, help="Annotator name → data/labels/<name>.jsonl")
@click.option("--all-traces", is_flag=True,
              help="Label every trace instead of the shared sample (not recommended).")
def main(make_sample, seed, benchmarks, annotator, all_traces):
    """Build the shared sample, or label it as a given annotator."""
    bms = [b.strip() for b in benchmarks.split(",")] if benchmarks else None

    if make_sample is not None:
        sample = _make_sample(make_sample, seed, bms)
        LABELS_DIR.mkdir(parents=True, exist_ok=True)
        SAMPLE_FILE.write_text(json.dumps(
            {"trace_ids": sample, "seed": seed, "n": len(sample),
             "created": datetime.now(timezone.utc).isoformat()}, indent=2), encoding="utf-8")
        print(f"Shared sample of {len(sample)} traces -> {SAMPLE_FILE}")
        print("Both annotators must label THIS sample for a valid kappa.")
        return

    if not annotator:
        print("Specify --annotator NAME (or --make-sample N first).")
        sys.exit(1)

    by_id = trace_by_id(bms)
    if all_traces:
        target_ids = list(by_id)
    else:
        if not SAMPLE_FILE.exists():
            print("No shared sample. Run --make-sample N first (or use --all-traces).")
            sys.exit(1)
        target_ids = json.loads(SAMPLE_FILE.read_text(encoding="utf-8"))["trace_ids"]

    done = _load_done(annotator)
    todo = [tid for tid in target_ids if tid not in done and tid in by_id]
    print(f"\nAnnotator: {annotator} | sample={len(target_ids)} | "
          f"done={len(done)} | remaining={len(todo)}")
    if not todo:
        print("Nothing to label. (All done, or sample/results mismatch.)")
        return
    print("Follow docs/labeling-guide.md. ARIA's prediction is hidden on purpose.")

    for i, tid in enumerate(todo, 1):
        _display(by_id[tid], i, len(todo))
        label, notes = _ask(i, len(todo))
        if label == "__quit__":
            print("\nStopped — progress saved. Re-run to resume.")
            break
        if label == "__skip__":
            continue
        _append_label(annotator, {
            "trace_id": tid,
            "benchmark": by_id[tid].benchmark,
            "label": label,
            "notes": notes or None,
            "annotator": annotator,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        print(f"  saved: {tid} -> {label}")

    total_done = len(_load_done(annotator))
    print(f"\n{annotator}: {total_done}/{len(target_ids)} labeled.")
    print("When both annotators finish: python scripts/compute_kappa.py "
          f"--a {annotator} --b <other>")


if __name__ == "__main__":
    main()
