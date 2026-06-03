#!/usr/bin/env python3
"""Run validation for the compiled Diagnostician in small batches.

The Groq free tier allows ~6 calls/minute. Run one batch at a time,
optionally with a different API key each time, then combine at the end.

Usage:
  # Run example index 0 (first example)
  python scripts/validate_diagnostician.py --index 0 --api-key gsk_xxx

  # Run examples 0 to 4 (5 examples)
  python scripts/validate_diagnostician.py --start 0 --end 4 --api-key gsk_xxx

  # After running all batches, combine results + full metrics report
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

DATA_DIR     = Path("data/synthetic")
COMPILED_DIR = Path("data/compiled")
VAL_DIR      = COMPILED_DIR / "val"

CLASSES = [
    "prompt_drift",
    "tool_misuse",
    "context_overflow",
    "goal_misalignment",
    "hallucination_loop",
    "none",
]


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
        try:
            confidence = float(getattr(pred, "confidence", "0.5"))
        except (ValueError, TypeError):
            confidence = 0.5
        reasoning = getattr(pred, "reasoning", "")
        return {
            "gold": gold,
            "predicted": predicted,
            "correct": correct,
            "confidence": confidence,
            "reasoning": reasoning[:200],
            "error": None,
        }
    except Exception as exc:
        return {
            "gold": gold,
            "predicted": None,
            "correct": False,
            "confidence": 0.0,
            "reasoning": "",
            "error": str(exc),
        }


@click.command()
@click.option("--start",   default=None, type=int, help="Start index (inclusive)")
@click.option("--end",     default=None, type=int, help="End index (inclusive)")
@click.option("--index",   default=None, type=int, help="Single example index")
@click.option("--api-key", default=None, help="Groq API key (overrides .env)")
@click.option("--delay",   default=12, show_default=True, type=float,
              help="Seconds between calls (keeps TPM under limit)")
@click.option("--combine", is_flag=True, help="Combine saved results and print full metrics report")
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

    if index is not None:
        start = index
        end = index
    if start is None or end is None:
        print("Provide --index N  or  --start N --end N  or  --combine")
        sys.exit(1)
    if start > end:
        print(f"--start ({start}) must be <= --end ({end})")
        sys.exit(1)

    key = api_key
    if not key:
        s = get_settings()
        key = s.groq_api_key
    if not key:
        print("No API key. Pass --api-key or set GROQ_API_KEY in .env")
        sys.exit(1)

    s = get_settings()
    lm = build_lm(api_key=key, model=f"groq/{s.groq_model}")
    dspy.configure(lm=lm)

    print("Loading validation set...")
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
    print(f"\nRunning examples {start}-{end} ({len(indices)} calls, ~{len(indices) * delay:.0f}s)\n")

    for i, idx in enumerate(indices):
        ex = valset[idx]
        print(f"  [{i + 1}/{len(indices)}] example {idx} | gold={ex.failure_class} ", end="", flush=True)

        result = run_single(program, ex)
        result["example_index"] = idx

        out_path = VAL_DIR / f"val_{idx:03d}.json"
        out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

        status = "OK" if result["correct"] else "WRONG"
        if result["error"]:
            print(f"ERROR: {result['error'][:80]}")
        else:
            conf = result.get("confidence", 0.0)
            print(f"pred={result['predicted']} [{status}] conf={conf:.2f}")

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

    all_results = [json.loads(f.read_text(encoding="utf-8")) for f in result_files]
    errors  = [r for r in all_results if r.get("error")]
    results = [r for r in all_results if not r.get("error")]
    total   = len(results)
    correct = sum(1 for r in results if r["correct"])

    SEP  = "-" * 62
    SEP2 = "=" * 62

    # ── Per-class precision / recall / F1 ────────────────────────
    tp: dict[str, int] = {c: 0 for c in CLASSES}
    fp: dict[str, int] = {c: 0 for c in CLASSES}
    fn: dict[str, int] = {c: 0 for c in CLASSES}

    for r in results:
        g = r["gold"]
        p = r["predicted"] or "unknown"
        if g == p:
            tp[g] = tp.get(g, 0) + 1
        else:
            fn[g] = fn.get(g, 0) + 1
            fp[p] = fp.get(p, 0) + 1

    print(f"\n{SEP2}")
    print(f"  ARIA DSPy Diagnostician -- Validation Report")
    print(f"  {total} examples evaluated  |  {len(errors)} API errors skipped")
    print(SEP2)

    print(f"\n  {'Class':<22}  {'Prec':>6}  {'Recall':>7}  {'F1':>6}  {'TP':>4}  {'FP':>4}  {'FN':>4}")
    print(f"  {SEP}")
    for cls in CLASSES:
        t   = tp.get(cls, 0)
        fp_ = fp.get(cls, 0)
        fn_ = fn.get(cls, 0)
        prec   = t / (t + fp_)  if (t + fp_) > 0 else 0.0
        recall = t / (t + fn_)  if (t + fn_) > 0 else 0.0
        f1     = 2 * prec * recall / (prec + recall) if (prec + recall) > 0 else 0.0
        print(f"  {cls:<22}  {prec:>5.1%}  {recall:>6.1%}  {f1:>5.1%}  {t:>4}  {fp_:>4}  {fn_:>4}")

    print(f"  {SEP}")
    print(f"  Overall accuracy: {correct}/{total} = {correct/total:.1%}")

    # ── Confusion matrix ──────────────────────────────────────────
    confusion: dict[str, dict[str, int]] = {g: {p: 0 for p in CLASSES} for g in CLASSES}
    for r in results:
        g = r["gold"]
        p = r["predicted"] or "unknown"
        if g in confusion and p in confusion[g]:
            confusion[g][p] += 1

    print(f"\n  CONFUSION MATRIX  (rows=gold, cols=predicted)")
    short = [c[:8] for c in CLASSES]
    print("  " + " " * 22 + "  ".join(f"{s:>8}" for s in short))
    print(f"  {SEP}")
    for g in CLASSES:
        row = f"  {g:<22}" + "  ".join(f"{confusion[g].get(p, 0):>8}" for p in CLASSES)
        print(row)

    # ── Confidence distribution ───────────────────────────────────
    print(f"\n  CONFIDENCE DISTRIBUTION")
    print(f"  {SEP}")
    by_conf: dict[str, list[float]] = {}
    for r in results:
        by_conf.setdefault(r["gold"], []).append(float(r.get("confidence", 0.5)))

    for cls in CLASSES:
        vals = by_conf.get(cls, [])
        if not vals:
            continue
        avg  = sum(vals) / len(vals)
        high = sum(1 for v in vals if v >= 0.8)
        low  = sum(1 for v in vals if v < 0.5)
        bar  = "#" * int(avg * 20)
        print(f"  {cls:<22}  avg={avg:.2f} [{bar:<20}]  high={high}  low={low}")

    # ── Hard pair focus ───────────────────────────────────────────
    hl_as_gm = confusion.get("hallucination_loop", {}).get("goal_misalignment", 0)
    gm_as_hl = confusion.get("goal_misalignment",  {}).get("hallucination_loop", 0)
    total_hard = hl_as_gm + gm_as_hl

    print(f"\n  HARD PAIR: hallucination_loop <-> goal_misalignment")
    print(f"  {SEP}")
    print(f"  hallucination_loop predicted as goal_misalignment : {hl_as_gm}")
    print(f"  goal_misalignment  predicted as hallucination_loop: {gm_as_hl}")
    if total_hard == 0:
        print("  Result: 0 confusions -- boundary held.")
    elif total_hard <= 3:
        print(f"  Result: {total_hard} confusions -- minor overlap, acceptable.")
    else:
        print(f"  Result: {total_hard} confusions -- WARNING: boundary is blurring.")

    print(f"\n{SEP2}\n")

    # ── Save JSON summary ─────────────────────────────────────────
    per_class_out = {}
    for cls in CLASSES:
        t   = tp.get(cls, 0)
        fp_ = fp.get(cls, 0)
        fn_ = fn.get(cls, 0)
        prec   = t / (t + fp_)  if (t + fp_) > 0 else 0.0
        recall = t / (t + fn_)  if (t + fn_) > 0 else 0.0
        f1     = 2 * prec * recall / (prec + recall) if (prec + recall) > 0 else 0.0
        per_class_out[cls] = {
            "precision": round(prec, 4),
            "recall":    round(recall, 4),
            "f1":        round(f1, 4),
            "tp": t, "fp": fp_, "fn": fn_,
        }

    summary_path = COMPILED_DIR / "val_summary.json"
    summary_path.write_text(
        json.dumps({
            "total":    total,
            "correct":  correct,
            "accuracy": round(correct / total, 4),
            "api_errors": len(errors),
            "per_class": per_class_out,
            "confusion_matrix": confusion,
            "hard_pair": {"hl_as_gm": hl_as_gm, "gm_as_hl": gm_as_hl},
        }, indent=2),
        encoding="utf-8",
    )
    print(f"  Full summary saved -> {summary_path}")


if __name__ == "__main__":
    main()
