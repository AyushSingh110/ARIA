#!/usr/bin/env python3
"""Compile the Diagnostician DSPy program using BootstrapFewShot.

Steps:
  1. Load synthetic training data from data/synthetic/
  2. Split train/val (80/20)
  3. Define accuracy metric: failure_class prediction matches gold label
  4. Run BootstrapFewShot compilation
  5. Save compiled program to data/compiled/diagnostician.json

Prerequisites:
  - Run scripts/generate_synthetic_data.py first
  - Set GROQ_API_KEY in .env

Run: python scripts/compile_diagnostician.py [--max-demos N] [--train-size N]
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import click

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

DATA_DIR = Path("data/synthetic")
COMPILED_DIR = Path("data/compiled")


def load_examples(per_class_limit: int) -> list:
    import dspy
    examples = []
    for fpath in sorted(DATA_DIR.glob("*.jsonl")):
        count = 0
        with fpath.open(encoding="utf-8") as f:
            for line in f:
                if count >= per_class_limit:
                    break
                rec = json.loads(line)
                examples.append(
                    dspy.Example(
                        task_description=rec["task_description"],
                        observer_flags=rec["observer_flags"],
                        critic_scores=rec["critic_scores"],
                        trace_summary=rec["trace_summary"],
                        failure_class=rec["failure_class"],
                        failure_manifestation=rec["failure_manifestation"],
                    ).with_inputs(
                        "task_description", "observer_flags", "critic_scores", "trace_summary"
                    )
                )
                count += 1
    return examples


def diagnostic_metric(gold, pred, trace=None) -> bool:
    gold_class = getattr(gold, "failure_class", "").strip().lower()
    pred_class = getattr(pred, "failure_class", "").strip().lower()
    return gold_class == pred_class


@click.command()
@click.option("--max-demos", default=4, show_default=True, help="Max bootstrapped demos per class")
@click.option("--per-class-limit", default=40, show_default=True, help="Training examples per class loaded")
@click.option("--val-ratio", default=0.2, show_default=True, help="Fraction of data for validation")
def main(max_demos: int, per_class_limit: int, val_ratio: float) -> None:
    """Compile the Diagnostician DSPy program with BootstrapFewShot."""
    import dspy
    from dspy.teleprompt import BootstrapFewShot

    from aria.config import get_settings
    from aria.dspy_programs.diagnostician import DiagnosticProgram, build_lm

    if not DATA_DIR.exists() or not list(DATA_DIR.glob("*.jsonl")):
        print("No synthetic data found. Run scripts/generate_synthetic_data.py first.")
        sys.exit(1)

    s = get_settings()
    if not s.groq_api_key:
        print("GROQ_API_KEY not set.")
        sys.exit(1)

    lm = build_lm(api_key=s.groq_api_key, model=f"groq/{s.groq_model}")
    dspy.configure(lm=lm)

    print("Loading synthetic examples…")
    all_examples = load_examples(per_class_limit)
    random.shuffle(all_examples)
    split = int(len(all_examples) * (1 - val_ratio))
    trainset = all_examples[:split]
    valset = all_examples[split:]
    print(f"  train={len(trainset)}  val={len(valset)}")

    optimizer = BootstrapFewShot(
        metric=diagnostic_metric,
        max_bootstrapped_demos=max_demos,
        max_labeled_demos=max_demos,
    )

    print("Compiling… (this calls Groq API)")
    program = DiagnosticProgram()
    compiled = optimizer.compile(program, trainset=trainset)

    COMPILED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = COMPILED_DIR / "diagnostician.json"
    compiled.save(str(out_path))
    print(f"Compiled program saved → {out_path}")
    print(f"\nValidation set size: {len(valset)} examples")
    print("Run validation separately (avoids rate limits):")
    print("  python scripts/validate_diagnostician.py --start 0 --end 4 --api-key gsk_xxx")
    print("  python scripts/validate_diagnostician.py --combine")


if __name__ == "__main__":
    main()
