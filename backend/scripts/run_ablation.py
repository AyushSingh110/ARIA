#!/usr/bin/env python3
"""Run the frozen test set through one ablation config (Step 2.2).

Produces the A/B/C/D ablation table — the single strongest artifact in the
paper. Run ONE config per invocation (keeps the per-process model/program caches
clean), then aggregate:

  python scripts/run_ablation.py --config a
  python scripts/run_ablation.py --config b
  python scripts/run_ablation.py --config c
  python scripts/run_ablation.py --config d
  python scripts/run_ablation.py --report      # prints the A/B/C/D table

For each test trace we rebuild the diagnosis state from the STORED trace and
re-run only the parts a config can change:
  * Observer flags never vary across A–D, so the stored flags are reused.
  * Critic: config A uses the holistic critic (re-run); B/C/D reuse the stored
    requirement-aware (v2) scores — valid because their critic is identical.
  * Diagnostician is always re-run under the config (rules/DSPy/XGB/grounding).

Predictions are written to data/ablation/<config>.jsonl as {trace_id, gold, pred}
and the run is resumable (already-scored trace_ids are skipped).

NOTE: config A re-runs the Critic (LLM calls); keep Ollama up. B/C/D are cheap
(no Critic call). Requires a frozen test set (scripts/freeze_test_set.py).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import click

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

TEST_FP = Path("data/splits/test.jsonl")
ABL_DIR = Path("data/ablation")
CONFIGS = ["a", "b", "c", "d"]


def _load_gold() -> dict[str, str]:
    if not TEST_FP.exists():
        print(f"ERROR: {TEST_FP} not found. Freeze a test set first.")
        sys.exit(1)
    return {json.loads(l)["trace_id"]: json.loads(l)["gold_label"]
            for l in TEST_FP.read_text(encoding="utf-8").splitlines() if l.strip()}


def _rebuild_state(trace):
    """Reconstruct a diagnosis state from a stored trace (raw result dict)."""
    from aria.state import make_initial_state
    r = trace.raw
    state = make_initial_state(task_description=trace.task,
                               task_class=trace.task_class or "general")
    state["executor_trace"] = r.get("trace") or []
    state["executor_output"] = trace.executor_output
    state["executor_turn_count"] = r.get("executor_turn_count", 0)
    # Observer output is config-invariant → reuse stored flags/drift.
    state["observer_flags"] = trace.observer_flags
    state["anomaly_detected"] = r.get("anomaly_detected", False)
    state["anomaly_severity"] = r.get("anomaly_severity", 0.0)
    state["drift_scores"] = r.get("drift_scores") or []
    # Requirement fields the Diagnostician reads (from stored v2 critic scores).
    cs = trace.critic_scores or {}
    state["critic_scores"] = cs or None
    state["requirement_checklist"] = cs.get("requirement_checklist", [])
    state["requirements_satisfied"] = cs.get("requirements_satisfied", [])
    state["requirement_satisfaction"] = cs.get("requirement_satisfaction", 0.0)
    return state


def _predict_one(trace) -> str:
    """Run Critic (config-dependent) + Diagnostician under the active ablation."""
    from aria.config.ablation import get_ablation
    from aria.agents.critic import critic_node
    from aria.agents.diagnostician import diagnostician_node

    state = _rebuild_state(trace)
    abl = get_ablation()
    # Config A (holistic critic) — re-run the Critic to get holistic scores.
    # B/C/D reuse the stored v2 scores already seeded into the state.
    if not abl.critic_v2 or not state.get("critic_scores"):
        state.update(critic_node(state))
        cs = state.get("critic_scores") or {}
        state["requirement_checklist"] = cs.get("requirement_checklist", [])
        state["requirements_satisfied"] = cs.get("requirements_satisfied", [])
        state["requirement_satisfaction"] = cs.get("requirement_satisfaction", 0.0)
    state.update(diagnostician_node(state))
    return (state.get("failure_class") or "none")


def _run_config(config: str) -> None:
    # Ablation must be set BEFORE importing the agent nodes (caches read env once).
    os.environ["ARIA_ABLATION"] = config
    from aria.config.ablation import get_ablation
    get_ablation.cache_clear()
    from aria.eval.dataset import trace_by_id

    gold = _load_gold()
    by_id = trace_by_id()
    ABL_DIR.mkdir(parents=True, exist_ok=True)
    out_fp = ABL_DIR / f"{config}.jsonl"

    done = set()
    if out_fp.exists():
        for l in out_fp.read_text(encoding="utf-8").splitlines():
            if l.strip():
                done.add(json.loads(l)["trace_id"])

    todo = [t for t in gold if t in by_id and t not in done]
    print(f"\nAblation {config.upper()} ({get_ablation().name}) | "
          f"test={len(gold)} done={len(done)} todo={len(todo)}")
    with out_fp.open("a", encoding="utf-8") as f:
        for i, tid in enumerate(todo, 1):
            try:
                pred = _predict_one(by_id[tid])
            except Exception as exc:
                print(f"  [{i}/{len(todo)}] {tid} ERROR {str(exc)[:60]}")
                continue
            f.write(json.dumps({"trace_id": tid, "gold": gold[tid], "pred": pred}) + "\n")
            f.flush()
            print(f"  [{i}/{len(todo)}] {tid}: {pred}")
    print(f"Wrote -> {out_fp}")


def _report() -> None:
    from aria.eval import metrics as M
    gold = _load_gold()
    print(f"\n{'=' * 60}\n  ABLATION TABLE (frozen test, N={len(gold)})\n{'=' * 60}")
    print(f"  {'config':<8}{'accuracy':>22}{'macro-F1':>22}")
    for c in CONFIGS:
        fp = ABL_DIR / f"{c}.jsonl"
        if not fp.exists():
            print(f"  {c.upper():<8}{'(not run)':>22}")
            continue
        rows = [json.loads(l) for l in fp.read_text(encoding="utf-8").splitlines() if l.strip()]
        g = [r["gold"] for r in rows]
        p = [r["pred"] for r in rows]
        acc, alo, ahi = M.bootstrap_ci(g, p, M.accuracy)
        mf1, flo, fhi = M.bootstrap_ci(g, p, M.macro_f1)
        print(f"  {c.upper():<8}{f'{acc:.1%} [{alo:.0%},{ahi:.0%}]':>22}"
              f"{f'{mf1:.2f} [{flo:.2f},{fhi:.2f}]':>22}")
    print(f"{'=' * 60}")


@click.command()
@click.option("--config", type=click.Choice(CONFIGS), default=None,
              help="Run this single ablation config over the frozen test set.")
@click.option("--report", "report", is_flag=True, help="Print the A/B/C/D table.")
def main(config, report):
    """Run one ablation config, or print the aggregated table."""
    if report:
        _report()
        return
    if not config:
        print("Specify --config a|b|c|d (one per invocation), or --report.")
        sys.exit(1)
    _run_config(config)
    print(f"\nNext: run the other configs, then: python scripts/run_ablation.py --report")


if __name__ == "__main__":
    main()
