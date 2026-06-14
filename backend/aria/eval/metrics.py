"""Classification metrics for small, imbalanced, labeled trace sets (Step 2.3).

Pure functions (no I/O) so they are unit-testable and reusable by the metrics
CLI and the ablation runner. Everything is implemented directly — no sklearn
dependency — so results are transparent and the repo stays light.

For ARIA's setting (N≈70, imbalanced classes) a single accuracy number is
misleading, so we report:
  * per-class precision / recall / F1 + support
  * macro-F1 (unweighted mean over classes — treats rare classes fairly)
  * a confusion matrix (which classes get confused — supports the
    mechanism/outcome thesis)
  * bootstrap 95% confidence intervals (with N≈70, "72%" is really "72% ± ~11%")
"""
from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass


def accuracy(gold: list[str], pred: list[str]) -> float:
    if not gold:
        return 0.0
    return sum(1 for g, p in zip(gold, pred) if g == p) / len(gold)


@dataclass
class ClassMetrics:
    label: str
    precision: float
    recall: float
    f1: float
    support: int          # number of gold instances of this class


def per_class_prf(gold: list[str], pred: list[str],
                  labels: list[str] | None = None) -> list[ClassMetrics]:
    """Precision/recall/F1/support per class."""
    labels = labels or sorted(set(gold) | set(pred))
    out: list[ClassMetrics] = []
    for lab in labels:
        tp = sum(1 for g, p in zip(gold, pred) if g == lab and p == lab)
        fp = sum(1 for g, p in zip(gold, pred) if g != lab and p == lab)
        fn = sum(1 for g, p in zip(gold, pred) if g == lab and p != lab)
        support = sum(1 for g in gold if g == lab)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        out.append(ClassMetrics(lab, prec, rec, f1, support))
    return out


def macro_f1(gold: list[str], pred: list[str],
             labels: list[str] | None = None) -> float:
    """Unweighted mean F1 over classes that have gold support."""
    cms = [c for c in per_class_prf(gold, pred, labels) if c.support > 0]
    return sum(c.f1 for c in cms) / len(cms) if cms else 0.0


def confusion_matrix(gold: list[str], pred: list[str],
                     labels: list[str] | None = None) -> tuple[list[str], list[list[int]]]:
    """Return (labels, matrix) where matrix[i][j] = #(gold=labels[i], pred=labels[j])."""
    labels = labels or sorted(set(gold) | set(pred))
    idx = {l: i for i, l in enumerate(labels)}
    m = [[0] * len(labels) for _ in labels]
    for g, p in zip(gold, pred):
        if g in idx and p in idx:
            m[idx[g]][idx[p]] += 1
    return labels, m


def bootstrap_ci(gold: list[str], pred: list[str], metric_fn,
                 n_boot: int = 2000, alpha: float = 0.05,
                 seed: int = 0) -> tuple[float, float, float]:
    """Bootstrap (point, lo, hi) for a metric over paired (gold, pred).

    Resamples trace indices with replacement n_boot times. Returns the metric on
    the full sample plus the [alpha/2, 1-alpha/2] percentile interval.
    """
    n = len(gold)
    point = metric_fn(gold, pred)
    if n == 0:
        return 0.0, 0.0, 0.0
    rng = random.Random(seed)
    stats = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        stats.append(metric_fn([gold[i] for i in idx], [pred[i] for i in idx]))
    stats.sort()
    lo = stats[int((alpha / 2) * n_boot)]
    hi = stats[min(n_boot - 1, int((1 - alpha / 2) * n_boot))]
    return point, lo, hi


def cohen_kappa(gold: list[str], pred: list[str]) -> float:
    """Chance-corrected agreement between two label sequences."""
    n = len(gold)
    if n == 0:
        return 0.0
    p_o = accuracy(gold, pred)
    g_marg, p_marg = Counter(gold), Counter(pred)
    labels = set(g_marg) | set(p_marg)
    p_e = sum((g_marg[l] / n) * (p_marg[l] / n) for l in labels)
    return (p_o - p_e) / (1 - p_e) if (1 - p_e) > 1e-9 else 1.0


def format_confusion(labels: list[str], matrix: list[list[int]]) -> str:
    """Compact text confusion matrix (rows = gold, cols = pred)."""
    short = [l[:8] for l in labels]
    head = "gold\\pred".ljust(12) + "".join(s.rjust(9) for s in short)
    lines = [head]
    for i, row in enumerate(matrix):
        lines.append(short[i].ljust(12) + "".join(str(v).rjust(9) for v in row))
    return "\n".join(lines)
