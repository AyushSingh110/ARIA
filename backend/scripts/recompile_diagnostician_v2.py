#!/usr/bin/env python3
"""Recompile the Diagnostician with the 5-field signature on REAL labeled data.

Training sources (human ground truth, not synthetic):
  1. data/realbench/results/  — 50 human-reviewed real agent runs
  2. data/gaia/results/       — GAIA runs with human_label (if reviewed)

Builds dspy.Example objects with the new requirement_summary field, runs
BootstrapFewShot, and saves to data/compiled/diagnostician_v2.json.
The pipeline can then load the v2 program (new signature) safely.

Run:
  cd backend
  python scripts/recompile_diagnostician_v2.py
  python scripts/recompile_diagnostician_v2.py --max-demos 6 --dry-run
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import click

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

REALBENCH_DIR = Path("data/realbench/results")
GAIA_DIR      = Path("data/gaia/results")
COMPILED_DIR  = Path("data/compiled")
OUT_PATH      = COMPILED_DIR / "diagnostician_v2.json"

# Human labels outside the taxonomy ("gap") cannot be used as training targets
VALID_CLASSES = {
    "prompt_drift", "tool_misuse", "context_overflow",
    "goal_misalignment", "hallucination_loop", "none",
}


def _requirement_summary_from(record: dict) -> str:
    cs = record.get("critic_scores") or {}
    reqs = (
        record.get("requirement_checklist")
        or cs.get("requirement_checklist")
        or []
    )
    sats = (
        record.get("requirements_satisfied")
        or cs.get("requirements_satisfied")
        or []
    )
    req_sat = (
        record.get("requirement_satisfaction")
        or cs.get("requirement_satisfaction")
        or 0.0
    )
    lines = [
        f"REQ: {r} [{'OK' if ok else 'MISS'}]"
        for r, ok in zip(reqs, sats)
    ]
    return "\n".join(lines) if lines else (
        f"requirement_satisfaction={float(req_sat):.2f} (no checklist available)"
    )


def load_labeled_examples() -> list:
    import dspy

    examples = []
    sources = {"realbench": 0, "gaia": 0}

    for src_name, src_dir, task_key in [
        ("realbench", REALBENCH_DIR, "task"),
        ("gaia", GAIA_DIR, "question"),
    ]:
        if not src_dir.exists():
            continue
        for f in sorted(src_dir.glob("*.json")):
            try:
                r = json.loads(f.read_text(encoding="utf-8"))
            except Exception:
                continue
            if r.get("run_error"):
                continue
            human = (r.get("human_label") or "").strip().lower()
            if not r.get("reviewed") or human not in VALID_CLASSES:
                continue

            examples.append(
                dspy.Example(
                    task_description=r.get(task_key) or r.get("task_description", ""),
                    observer_flags=json.dumps(r.get("observer_flags") or []),
                    critic_scores=json.dumps(r.get("critic_scores") or {}),
                    requirement_summary=_requirement_summary_from(r),
                    trace_summary=r.get("trace_summary", ""),
                    failure_class=human,
                    failure_manifestation=r.get("aria_manifestation") or "none",
                ).with_inputs(
                    "task_description", "observer_flags", "critic_scores",
                    "requirement_summary", "trace_summary",
                )
            )
            sources[src_name] += 1

    print(f"Loaded labeled examples: realbench={sources['realbench']}  gaia={sources['gaia']}")
    return examples


def diagnostic_metric(gold, pred, trace=None) -> bool:
    gold_class = getattr(gold, "failure_class", "").strip().lower()
    pred_class = getattr(pred, "failure_class", "").strip().lower()
    return gold_class == pred_class


@click.command()
@click.option("--max-demos", default=4, show_default=True, help="Max bootstrapped demos")
@click.option("--val-ratio", default=0.2, show_default=True, help="Validation fraction")
@click.option("--seed", default=42, show_default=True)
@click.option("--dry-run", is_flag=True, help="Show training set stats without compiling")
def main(max_demos: int, val_ratio: float, seed: int, dry_run: bool) -> None:
    """Recompile Diagnostician v2 on human-labeled real-world data."""
    import dspy
    from collections import Counter
    from dspy.teleprompt import BootstrapFewShot

    from aria.config import get_settings
    from aria.dspy_programs.diagnostician import DiagnosticProgram, build_lm

    examples = load_labeled_examples()
    if len(examples) < 20:
        print(f"Only {len(examples)} labeled examples — need 20+ for a useful compile.")
        print("Label more runs first (scripts/review_realbench.py).")
        if not dry_run:
            sys.exit(1)

    dist = Counter(e.failure_class for e in examples)
    print("Class distribution in training data:")
    for cls, n in dist.most_common():
        print(f"  {cls:<22} {n}")

    random.seed(seed)
    random.shuffle(examples)
    split = int(len(examples) * (1 - val_ratio))
    trainset, valset = examples[:split], examples[split:]
    print(f"train={len(trainset)}  val={len(valset)}")

    if dry_run:
        print("\n--dry-run: stopping before compilation.")
        return

    s = get_settings()
    if not s.groq_api_key:
        print("GROQ_API_KEY not set.")
        sys.exit(1)

    lm = build_lm(api_key=s.groq_api_key, model=f"groq/{s.groq_model}")
    dspy.configure(lm=lm)

    optimizer = BootstrapFewShot(
        metric=diagnostic_metric,
        max_bootstrapped_demos=max_demos,
        max_labeled_demos=max_demos,
    )

    print("\nCompiling on real labeled data... (calls Groq API)")
    compiled = optimizer.compile(DiagnosticProgram(), trainset=trainset)

    COMPILED_DIR.mkdir(parents=True, exist_ok=True)
    compiled.save(str(OUT_PATH))
    print(f"Compiled v2 program saved -> {OUT_PATH}")

    # Quick validation on held-out set
    print(f"\nValidating on {len(valset)} held-out examples...")
    hits = 0
    for ex in valset:
        try:
            pred = compiled(
                task_description=ex.task_description,
                observer_flags=ex.observer_flags,
                critic_scores=ex.critic_scores,
                requirement_summary=ex.requirement_summary,
                trace_summary=ex.trace_summary,
            )
            if diagnostic_metric(ex, pred):
                hits += 1
        except Exception as exc:
            print(f"  validation error: {exc}")
    if valset:
        print(f"Held-out accuracy: {hits}/{len(valset)} = {hits/len(valset)*100:.1f}%")


if __name__ == "__main__":
    main()
