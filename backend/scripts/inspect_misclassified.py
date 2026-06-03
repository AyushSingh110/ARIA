#!/usr/bin/env python3
"""Inspect misclassified examples from the DSPy validation run.

Focuses on the two failure patterns found in validation:
  A. goal_misalignment predicted as tool_misuse  (4 examples)
  B. hallucination_loop predicted as goal_misalignment  (2 examples)

For each misclassified example, shows:
  - Full task description
  - Full trace
  - Observer flags
  - Critic scores
  - DSPy confidence + reasoning
  - Manual question: Dataset problem or Ontology problem?

Outputs report to research/misclassification_analysis_v1.md

Run: python scripts/inspect_misclassified.py
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

DATA_DIR     = Path("data/synthetic")
COMPILED_DIR = Path("data/compiled")
VAL_DIR      = COMPILED_DIR / "val"
REPORT_DIR   = Path("../research")


def load_valset(per_class_limit: int = 40, val_ratio: float = 0.2, seed: int = 42) -> list[dict]:
    """Load valset as plain dicts (not DSPy examples) to preserve all fields."""
    all_examples = []
    for fpath in sorted(DATA_DIR.glob("*.jsonl")):
        count = 0
        with fpath.open(encoding="utf-8") as f:
            for line in f:
                if count >= per_class_limit:
                    break
                rec = json.loads(line)
                all_examples.append(rec)
                count += 1

    random.seed(seed)
    random.shuffle(all_examples)
    split = int(len(all_examples) * (1 - val_ratio))
    return all_examples[split:]


def load_val_results() -> list[dict]:
    if not VAL_DIR.exists():
        print("ERROR: No val results found. Run validate_diagnostician.py first.")
        sys.exit(1)
    results = []
    for f in sorted(VAL_DIR.glob("val_*.json")):
        r = json.loads(f.read_text(encoding="utf-8"))
        if not r.get("error"):
            results.append(r)
    return results


def fmt_example(ex: dict, result: dict, idx: int, label: str) -> str:
    flags  = json.loads(ex.get("observer_flags", "[]"))
    scores = json.loads(ex.get("critic_scores",  "{}"))
    flag_str = ", ".join(f["flag_type"] for f in flags) if flags else "none"

    sep = "-" * 62

    lines = [
        f"",
        f"  [{label}]  Example {idx + 1}",
        f"  {sep}",
        f"  Task class     : {ex.get('task_class', '?')}",
        f"  Task           : {ex.get('task_description', '?')}",
        f"  Gold label     : {result['gold']}",
        f"  Predicted      : {result['predicted']}",
        f"  Confidence     : {result.get('confidence', '?')}",
        f"  Manifestation  : {ex.get('failure_manifestation', '?')}",
        f"  Turns          : {ex.get('executor_turn_count', '?')}",
        f"  Obs flags      : {flag_str}",
        f"  Critic scores  : corr={scores.get('correctness','?')}  "
          f"comp={scores.get('completeness','?')}  "
          f"eff={scores.get('efficiency','?')}  "
          f"pass={scores.get('pass_fail','?')}",
        f"",
        f"  Full trace:",
    ]

    for line in ex.get("trace_summary", "").split("\n"):
        lines.append(f"    {line}")

    lines += [
        f"",
        f"  DSPy reasoning:",
    ]
    for line in result.get("reasoning", "(none)").split("."):
        line = line.strip()
        if line:
            lines.append(f"    {line}.")

    lines += [
        f"",
        f"  MANUAL QUESTION:",
        f"  Reading only the trace above, could a human reasonably label",
        f"  this as '{result['predicted']}' instead of '{result['gold']}'?",
        f"  [ ] Yes -> Ontology problem (genuine overlap between classes)",
        f"  [ ] No  -> Dataset problem (trace doesn't encode the class clearly enough)",
        f"",
    ]

    return "\n".join(lines)


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading valset and results...")
    valset  = load_valset()
    results = load_val_results()

    if len(results) != len(valset):
        print(f"  WARNING: {len(results)} results but {len(valset)} valset examples.")
        print("  Some batches may be missing. Run all batches then re-run this script.")

    # Build index: example_index -> (valset_example, result)
    indexed = {}
    for r in results:
        idx = r.get("example_index")
        if idx is not None and idx < len(valset):
            indexed[idx] = (valset[idx], r)

    # Find target pairs
    gm_as_tm  = [(i, ex, r) for i, (ex, r) in indexed.items()
                 if r["gold"] == "goal_misalignment" and r["predicted"] == "tool_misuse"]
    hl_as_gm  = [(i, ex, r) for i, (ex, r) in indexed.items()
                 if r["gold"] == "hallucination_loop" and r["predicted"] == "goal_misalignment"]

    SEP2 = "=" * 62

    output_lines = [
        "ARIA Misclassification Analysis -- Research Cycle 0.75",
        SEP2,
        f"goal_misalignment predicted as tool_misuse : {len(gm_as_tm)} examples",
        f"hallucination_loop predicted as goal_misalignment: {len(hl_as_gm)} examples",
        "",
        "PURPOSE: Determine for each confusion whether the root cause",
        "is a Dataset Problem or an Ontology Problem.",
        "",
        "DATASET PROBLEM  : trace does not clearly encode the class.",
        "                   Fix: improve synthetic data generator.",
        "ONTOLOGY PROBLEM : classes genuinely overlap at semantic level.",
        "                   Fix: revise taxonomy (possibly merge or add layer).",
        "",
    ]

    # ── Section A: goal_misalignment → tool_misuse ────────────────
    output_lines += [
        SEP2,
        "SECTION A: goal_misalignment predicted as tool_misuse",
        SEP2,
        "",
        "Context: goal_misalignment = agent solves wrong objective (silent failure).",
        "         tool_misuse       = agent calls wrong/broken tool (observable failure).",
        "",
        "Mentor hypothesis: these may overlap because tool_misuse can be the",
        "MECHANISM that causes goal_misalignment as an OUTCOME.",
        "",
    ]

    for i, (vidx, ex, r) in enumerate(gm_as_tm):
        output_lines.append(fmt_example(ex, r, i, f"A{i+1}"))

    # ── Section B: hallucination_loop → goal_misalignment ─────────
    output_lines += [
        SEP2,
        "SECTION B: hallucination_loop predicted as goal_misalignment",
        SEP2,
        "",
        "Context: hallucination_loop  = agent invents facts without tool grounding.",
        "         goal_misalignment   = agent solves proxy objective.",
        "",
        "Both classes: 0 observer flags, pass_fail=False.",
        "Key distinguisher: LLM turn ratio (hallucination=high, misalignment=low).",
        "",
    ]

    for i, (vidx, ex, r) in enumerate(hl_as_gm):
        output_lines.append(fmt_example(ex, r, i, f"B{i+1}"))

    # ── Summary questions ─────────────────────────────────────────
    output_lines += [
        SEP2,
        "SUMMARY QUESTIONS FOR MENTOR MEETING",
        SEP2,
        "",
        "After reading all examples above, answer:",
        "",
        "Q1. In Section A: are the goal_misalignment examples genuinely",
        "    ambiguous with tool_misuse, or do they clearly encode a wrong",
        "    objective without tool errors?",
        "",
        "Q2. Does the taxonomy need a two-layer structure?",
        "    Layer 1 (mechanisms): tool_misuse, prompt_drift,",
        "                          context_overflow, hallucination_loop",
        "    Layer 2 (outcomes):   goal_misalignment, task_failure,",
        "                          partial_completion",
        "",
        "Q3. Should goal_misalignment be removed as a sibling class and",
        "    reframed as an outcome that OTHER classes can produce?",
        "",
        "Q4. For hallucination_loop: is 40% recall acceptable for v1,",
        "    or does HallucinationBench need to be built before publication?",
        "",
        SEP2,
    ]

    full_output = "\n".join(output_lines)

    # Print to terminal
    print(full_output.encode("ascii", errors="replace").decode("ascii"))

    # Save to file (full unicode)
    report_path = REPORT_DIR / "misclassification_analysis_v1.md"
    report_path.write_text(full_output, encoding="utf-8")
    print(f"\nFull report saved -> {report_path}")

    # Print counts
    print(f"\nSummary:")
    print(f"  goal_misalignment -> tool_misuse : {len(gm_as_tm)} examples")
    print(f"  hallucination_loop -> goal_misalignment: {len(hl_as_gm)} examples")
    print(f"  Total inspected: {len(gm_as_tm) + len(hl_as_gm)}")


if __name__ == "__main__":
    main()
