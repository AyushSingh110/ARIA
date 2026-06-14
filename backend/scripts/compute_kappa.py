#!/usr/bin/env python3
"""Inter-rater agreement for two annotators (Step 1.4).

Reads data/labels/<a>.jsonl and data/labels/<b>.jsonl, restricts to the traces
both labeled, and reports:
  * raw agreement %
  * Cohen's kappa (chance-corrected agreement) — target > 0.6
  * the per-pair confusion (which class pairs they disagree on)
  * the list of disagreements for adjudication

Then, after the two of you adjudicate together, build the gold labels:
  python scripts/compute_kappa.py --a ayush --b classmate --adjudicate
  → writes data/labels/gold.jsonl  (agreements auto-accepted; disagreements
    resolved interactively). This gold file is what freeze_test_set.py consumes.

Cohen's kappa is implemented directly (no sklearn dependency):
    kappa = (p_o - p_e) / (1 - p_e)
  p_o = observed agreement; p_e = agreement expected by chance from each
  annotator's marginal label frequencies.
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import click

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

LABELS_DIR = Path("data/labels")


def _load(annotator: str) -> dict[str, str]:
    fp = LABELS_DIR / f"{annotator}.jsonl"
    if not fp.exists():
        print(f"ERROR: {fp} not found. Label first: "
              f"python scripts/label_traces.py --annotator {annotator}")
        sys.exit(1)
    out: dict[str, str] = {}
    for line in fp.read_text(encoding="utf-8").splitlines():
        try:
            rec = json.loads(line)
            out[rec["trace_id"]] = rec["label"]   # last label for an id wins
        except Exception:
            pass
    return out


def cohen_kappa(pairs: list[tuple[str, str]]) -> tuple[float, float, float]:
    """Return (kappa, observed_agreement, expected_agreement)."""
    n = len(pairs)
    if n == 0:
        return 0.0, 0.0, 0.0
    p_o = sum(1 for a, b in pairs if a == b) / n
    a_marg = Counter(a for a, _ in pairs)
    b_marg = Counter(b for _, b in pairs)
    labels = set(a_marg) | set(b_marg)
    p_e = sum((a_marg[l] / n) * (b_marg[l] / n) for l in labels)
    kappa = (p_o - p_e) / (1 - p_e) if (1 - p_e) > 1e-9 else 1.0
    return kappa, p_o, p_e


def _report(a_name, b_name, a_lab, b_lab) -> list[tuple[str, str, str]]:
    shared = sorted(set(a_lab) & set(b_lab))
    if not shared:
        print("No overlapping traces — both annotators must label the SAME sample.")
        sys.exit(1)
    pairs = [(a_lab[t], b_lab[t]) for t in shared]
    kappa, p_o, p_e = cohen_kappa(pairs)

    print(f"\n{'=' * 56}")
    print(f"  Inter-rater agreement: {a_name} vs {b_name}")
    print(f"{'=' * 56}")
    print(f"  Traces labeled by both : {len(shared)}")
    print(f"  Raw agreement          : {p_o:.1%}")
    print(f"  Expected (chance)      : {p_e:.1%}")
    print(f"  Cohen's kappa          : {kappa:.3f}  "
          f"({'OK >0.6' if kappa > 0.6 else 'LOW — refine guide & relabel'})")

    disagreements = [(t, a_lab[t], b_lab[t]) for t in shared if a_lab[t] != b_lab[t]]
    if disagreements:
        print(f"\n  Disagreements ({len(disagreements)}):")
        pair_counts = Counter((min(a, b), max(a, b)) for _, a, b in disagreements)
        for (x, y), c in pair_counts.most_common():
            print(f"    {x} <-> {y}: {c}")
    print(f"{'=' * 56}")
    return disagreements


def _adjudicate(a_name, b_name, a_lab, b_lab) -> None:
    from aria.eval.dataset import VALID_LABELS, trace_by_id
    shared = sorted(set(a_lab) & set(b_lab))
    by_id = trace_by_id()
    gold_fp = LABELS_DIR / "gold.jsonl"
    gold: list[dict] = []

    auto = [t for t in shared if a_lab[t] == b_lab[t]]
    disag = [t for t in shared if a_lab[t] != b_lab[t]]
    for t in auto:
        gold.append({"trace_id": t, "label": a_lab[t], "source": "agreement"})

    print(f"\n{len(auto)} agreements auto-accepted. Adjudicate {len(disag)} disagreements:")
    menu = "  ".join(f"{i+1}.{l}" for i, l in enumerate(VALID_LABELS))
    for i, t in enumerate(disag, 1):
        tr = by_id.get(t)
        print(f"\n[{i}/{len(disag)}] {t}")
        if tr:
            print(f"  Task: {tr.task[:160]}")
            print(f"  Trace: {(tr.trace_summary or '')[:200]}")
        print(f"  {a_name}={a_lab[t]}   {b_name}={b_lab[t]}")
        print(f"  {menu}")
        while True:
            raw = input("  Final gold label (number/name): ").strip().lower()
            label = None
            if raw.isdigit() and 1 <= int(raw) <= len(VALID_LABELS):
                label = VALID_LABELS[int(raw) - 1]
            elif raw in VALID_LABELS:
                label = raw
            if label:
                gold.append({"trace_id": t, "label": label, "source": "adjudicated",
                             "a": a_lab[t], "b": b_lab[t]})
                break
            print("  invalid")

    for g in gold:
        g["adjudicated_at"] = datetime.now(timezone.utc).isoformat()
    gold_fp.write_text("\n".join(json.dumps(g, ensure_ascii=False) for g in gold) + "\n",
                       encoding="utf-8")
    print(f"\nWrote {len(gold)} gold labels -> {gold_fp}")
    print("Next: python scripts/freeze_test_set.py --gold data/labels/gold.jsonl")


@click.command()
@click.option("--a", "a_name", required=True, help="First annotator name")
@click.option("--b", "b_name", required=True, help="Second annotator name")
@click.option("--adjudicate", is_flag=True,
              help="Interactively resolve disagreements and write gold.jsonl")
def main(a_name, b_name, adjudicate):
    """Report kappa, or adjudicate to gold labels."""
    a_lab, b_lab = _load(a_name), _load(b_name)
    _report(a_name, b_name, a_lab, b_lab)
    if adjudicate:
        _adjudicate(a_name, b_name, a_lab, b_lab)


if __name__ == "__main__":
    main()
