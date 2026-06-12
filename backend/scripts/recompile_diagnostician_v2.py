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
  python scripts/recompile_diagnostician_v2.py               # full compile + validate held-out
  python scripts/recompile_diagnostician_v2.py --validate-all # validate ALL 48 examples (resumable)
  python scripts/recompile_diagnostician_v2.py --validate-only # held-out only, skip recompile
  python scripts/recompile_diagnostician_v2.py --dry-run      # stats preview, no API calls
"""
from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

import click

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

REALBENCH_DIR  = Path("data/realbench/results")
GAIA_DIR       = Path("data/gaia/results")
COMPILED_DIR   = Path("data/compiled")
OUT_PATH       = COMPILED_DIR / "diagnostician_v2.json"
CHECKPOINT     = COMPILED_DIR / "validate_all_checkpoint.json"

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


def _run_validation(compiled, examples: list, label: str, checkpoint_path: Path | None = None) -> None:
    """Validate compiled program against examples, with checkpoint/resume support.

    If checkpoint_path is given, already-evaluated indices are skipped so the
    run can be aborted and resumed after swapping API keys.
    """
    # Load checkpoint (maps str(index) -> {"gold": str, "pred": str, "correct": bool})
    ckpt: dict = {}
    if checkpoint_path and checkpoint_path.exists():
        try:
            ckpt = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            done = sum(1 for v in ckpt.values() if v.get("done"))
            print(f"  Resuming: {done}/{len(examples)} already evaluated (checkpoint loaded)")
        except Exception:
            ckpt = {}

    hits = sum(1 for v in ckpt.values() if v.get("correct"))
    total_done = sum(1 for v in ckpt.values() if v.get("done"))

    for i, ex in enumerate(examples):
        key = str(i)
        if ckpt.get(key, {}).get("done"):
            status = "correct" if ckpt[key]["correct"] else "wrong"
            print(f"  [{i+1:>2}/{len(examples)}] skip (cached: {status}) — gold={ckpt[key]['gold']}")
            continue

        retries = 0
        success = False
        while retries < 5:
            try:
                pred = compiled(
                    task_description=ex.task_description,
                    observer_flags=ex.observer_flags,
                    critic_scores=ex.critic_scores,
                    requirement_summary=ex.requirement_summary,
                    trace_summary=ex.trace_summary,
                )
                correct = diagnostic_metric(ex, pred)
                pred_class = getattr(pred, "failure_class", "?").strip().lower()
                if correct:
                    hits += 1
                total_done += 1

                marker = "OK" if correct else "MISS"
                print(f"  [{i+1:>2}/{len(examples)}] {marker}  gold={ex.failure_class:<22} pred={pred_class}")

                if checkpoint_path:
                    ckpt[key] = {"gold": ex.failure_class, "pred": pred_class, "correct": correct, "done": True}
                    checkpoint_path.write_text(json.dumps(ckpt, indent=2), encoding="utf-8")

                success = True
                break

            except Exception as exc:
                msg = str(exc)
                if "rate_limit" in msg.lower() or "ratelimit" in msg.lower():
                    wait = 20 * (retries + 1)
                    print(f"  [{i+1:>2}/{len(examples)}] rate limit hit — waiting {wait}s then retrying...")
                    time.sleep(wait)
                    retries += 1
                else:
                    print(f"  [{i+1:>2}/{len(examples)}] error: {exc}")
                    break

        if not success:
            print(f"\n  [{i+1:>2}/{len(examples)}] ABORTED after retries exhausted.")
            print("  To resume:")
            print("  1. Update GROQ_API_KEY in backend/.env with a new key")
            print("  2. Rerun the same command — it will skip already-done examples")
            if checkpoint_path:
                print(f"  Checkpoint saved: {checkpoint_path}")
            sys.exit(1)

        # 8-second gap keeps TPM under the 12k/min Groq on_demand ceiling
        if i < len(examples) - 1:
            time.sleep(8)

    if examples:
        pct = hits / len(examples) * 100
        print(f"\n{label} accuracy: {hits}/{len(examples)} = {pct:.1f}%")

    # Clean up checkpoint after successful completion
    if checkpoint_path and checkpoint_path.exists():
        checkpoint_path.unlink(missing_ok=True)
        print("  Checkpoint cleared (run complete).")


@click.command()
@click.option("--max-demos", default=4, show_default=True, help="Max bootstrapped demos")
@click.option("--val-ratio", default=0.2, show_default=True, help="Validation fraction")
@click.option("--seed", default=42, show_default=True)
@click.option("--dry-run", is_flag=True, help="Show training set stats without compiling")
@click.option("--validate-only", is_flag=True, help="Skip compilation; load existing v2 JSON and run held-out validation")
@click.option("--validate-all", is_flag=True, help="Validate against ALL 48 examples (resumable — safe to abort + rerun)")
def main(max_demos: int, val_ratio: float, seed: int, dry_run: bool, validate_only: bool, validate_all: bool) -> None:
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

    # Reload .env so a key swap before resume is picked up without re-import
    from dotenv import load_dotenv
    load_dotenv(override=True)

    s = get_settings()
    if not s.groq_api_key:
        print("GROQ_API_KEY not set in .env")
        sys.exit(1)

    lm = build_lm(api_key=s.groq_api_key, model=f"groq/{s.groq_model}")
    dspy.configure(lm=lm)

    # Tell litellm to auto-retry rate limit errors during bootstrap
    # (handles 429s transparently so the compile doesn't drop examples)
    try:
        import litellm
        litellm.num_retries = 6
        litellm.request_timeout = 120
    except Exception:
        pass

    if validate_only or validate_all:
        if not OUT_PATH.exists():
            print(f"No compiled program at {OUT_PATH}. Run without --validate-only/--validate-all first.")
            sys.exit(1)
        print(f"Loading existing program from {OUT_PATH}")
        compiled = DiagnosticProgram()
        compiled.load(str(OUT_PATH))
    else:
        optimizer = BootstrapFewShot(
            metric=diagnostic_metric,
            max_bootstrapped_demos=max_demos,
            max_labeled_demos=max_demos,
        )
        print("\nCompiling on real labeled data... (calls Groq API)")
        print("  Rate-limit retries enabled (litellm will auto-retry 429s).")
        print("  If bootstrap still fails, swap GROQ_API_KEY in .env and rerun — model saves on completion.\n")
        compiled = optimizer.compile(DiagnosticProgram(), trainset=trainset)
        COMPILED_DIR.mkdir(parents=True, exist_ok=True)
        compiled.save(str(OUT_PATH))
        print(f"Compiled v2 program saved -> {OUT_PATH}")

    if validate_all:
        print(f"\nValidating on ALL {len(examples)} examples (resumable — checkpoint: {CHECKPOINT})")
        print("  Safe to abort at any time. Rerun with --validate-all to continue from where you stopped.\n")
        _run_validation(compiled, examples, label=f"Full ({len(examples)} examples)", checkpoint_path=CHECKPOINT)
    else:
        print(f"\nValidating on {len(valset)} held-out examples...")
        _run_validation(compiled, valset, label=f"Held-out ({len(valset)} examples)", checkpoint_path=None)


if __name__ == "__main__":
    main()
