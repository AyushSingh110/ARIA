#!/usr/bin/env python3
"""Run validation for the compiled Diagnostician in small batches.

The Groq free tier allows ~6 calls/minute. Run one batch at a time,
optionally with a different API key each time, then combine at the end.

Usage:
  # Run example index 0 (first example)
  python scripts/validate_diagnostician.py --index 0 --api-key gsk_xxx

  # Run examples 0 to 4 (5 examples)
  python scripts/validate_diagnostician.py --start 0 --end 4 --api-key gsk_xxx

  # After running all batches, combine results
  python scripts/validate_diagnostician.py --combine

Results are saved to data/compiled/val/ as individual JSON files.
"""
from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

import click

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

DATA_DIR = Path("data/synthetic")
COMPILED_DIR = Path("data/compiled")
VAL_DIR = COMPILED_DIR / "val"


def load_valset(per_class_limit: int = 40, val_ratio: float = 0.2, seed: int = 42) -> list:
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

    random.seed(seed)
    random.shuffle(examples)
    split = int(len(examples) * (1 - val_ratio))
    return examples[split:]


def run_single(program, example) -> dict:
    gold = example.failure_class.strip().lower()
    try:
        pred = program(**example.inputs())
        predicted = getattr(pred, "failure_class", "").strip().lower()
        correct = gold == predicted
        return {
            "gold": gold,
            "predicted": predicted,
            "correct": correct,
            "error": None,
        }
    except Exception as exc:
        return {
            "gold": gold,
            "predicted": None,
            "correct": False,
            "error": str(exc),
        }


@click.command()
@click.option("--start", default=None, type=int, help="Start index (inclusive)")
@click.option("--end", default=None, type=int, help="End index (inclusive)")
@click.option("--index", default=None, type=int, help="Single example index (shorthand for --start N --end N)")
@click.option("--api-key", default=None, help="Groq API key (overrides .env)")
@click.option("--delay", default=12, show_default=True, type=float,
              help="Seconds to wait between calls (keeps TPM under limit)")
@click.option("--combine", is_flag=True, help="Combine all saved batch results and print final accuracy")
def main(start, end, index, api_key, delay, combine):
    """Validate compiled Diagnostician in rate-limit-safe batches."""

    if combine:
        _combine_results()
        return

    import dspy
    from aria.config import get_settings
    from aria.dspy_programs.diagnostician import DiagnosticProgram, build_lm

    compiled_path = COMPILED_DIR / "diagnostician.json"
    if not compiled_path.exists():
        print("No compiled program found. Run compile_diagnostician.py first.")
        sys.exit(1)

    if not DATA_DIR.exists():
        print("No synthetic data. Run generate_synthetic_data.py first.")
        sys.exit(1)

    # Resolve index range
    if index is not None:
        start = index
        end = index
    if start is None or end is None:
        print("Provide --index N  or  --start N --end N  or  --combine")
        sys.exit(1)
    if start > end:
        print(f"--start ({start}) must be <= --end ({end})")
        sys.exit(1)

    # API key: CLI arg > .env
    key = api_key
    if not key:
        s = get_settings()
        key = s.groq_api_key
    if not key:
        print("No API key. Pass --api-key or set GROQ_API_KEY in .env")
        sys.exit(1)

    from aria.config import get_settings
    s = get_settings()
    lm = build_lm(api_key=key, model=f"groq/{s.groq_model}")
    dspy.configure(lm=lm)

    print("Loading validation set…")
    valset = load_valset()
    total = len(valset)
    print(f"  Total val examples: {total}")

    if start > total - 1:
        print(f"Index {start} out of range (max {total - 1})")
        sys.exit(1)
    end = min(end, total - 1)

    program = DiagnosticProgram()
    program.load(str(compiled_path))

    VAL_DIR.mkdir(parents=True, exist_ok=True)

    indices = list(range(start, end + 1))
    print(f"\nRunning examples {start}–{end} ({len(indices)} calls, ~{len(indices) * delay:.0f}s)\n")

    for i, idx in enumerate(indices):
        ex = valset[idx]
        print(f"  [{i + 1}/{len(indices)}] example {idx} | gold={ex.failure_class} ", end="", flush=True)

        result = run_single(program, ex)
        result["example_index"] = idx

        out_path = VAL_DIR / f"val_{idx:03d}.json"
        out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

        status = "✓" if result["correct"] else "✗"
        if result["error"]:
            print(f"ERROR: {result['error'][:80]}")
        else:
            print(f"pred={result['predicted']} {status}")

        if i < len(indices) - 1:
            time.sleep(delay)

    print(f"\nSaved to {VAL_DIR}/val_*.json")
    print("When all batches are done, run:  python scripts/validate_diagnostician.py --combine")


def _combine_results():
    if not VAL_DIR.exists():
        print("No results directory found. Run some batches first.")
        sys.exit(1)

    result_files = sorted(VAL_DIR.glob("val_*.json"))
    if not result_files:
        print("No result files found in data/compiled/val/")
        sys.exit(1)

    results = [json.loads(f.read_text(encoding="utf-8")) for f in result_files]
    total = len(results)
    correct = sum(1 for r in results if r["correct"])
    errors = sum(1 for r in results if r["error"])

    print(f"\n{'─' * 40}")
    print(f"Validation results ({total} examples)")
    print(f"{'─' * 40}")

    by_class: dict[str, list[bool]] = {}
    for r in results:
        cls = r["gold"]
        by_class.setdefault(cls, []).append(r["correct"])

    for cls, preds in sorted(by_class.items()):
        n_correct = sum(preds)
        print(f"  {cls:<22} {n_correct}/{len(preds)} = {n_correct/len(preds):.0%}")

    print(f"{'─' * 40}")
    print(f"  Overall accuracy:      {correct}/{total} = {correct/total:.0%}")
    if errors:
        print(f"  API errors (skipped): {errors}")
    print(f"{'─' * 40}\n")

    summary_path = COMPILED_DIR / "val_summary.json"
    summary_path.write_text(
        json.dumps({"total": total, "correct": correct, "errors": errors,
                    "accuracy": correct / total, "per_class": {
                        cls: {"correct": sum(v), "total": len(v)} for cls, v in by_class.items()
                    }}, indent=2),
        encoding="utf-8"
    )
    print(f"Summary saved → {summary_path}")


if __name__ == "__main__":
    main()
