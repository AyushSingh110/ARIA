"""Unified loader over every benchmark's result files.

Each benchmark runner writes per-trace JSON into ``data/<bench>/results/``, but
the field names drift (GAIA uses ``gaia_correct``/``question``; the newer
runners use ``answer_correct``/``task``). Every methodology script needs a
*single* consistent view of a trace, so this module normalises them into one
``Trace`` shape and exposes a stable, globally-unique ``trace_id``.

``trace_id`` = ``"<benchmark>/<result-file-stem>"`` — unique across benchmarks
and across executor-config variants (the config is encoded in the file stem),
and stable across re-runs, so labels/splits keyed on it never get orphaned.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional

DATA_ROOT = Path("data")

# Benchmark name -> results directory. Add new benchmarks here once.
RESULT_DIRS: dict[str, Path] = {
    "gsm8k":    DATA_ROOT / "gsm8k" / "results",
    "hotpotqa": DATA_ROOT / "hotpotqa" / "results",
    "2wiki":    DATA_ROOT / "2wiki" / "results",
    "gaia":     DATA_ROOT / "gaia" / "results",
    "taubench": DATA_ROOT / "taubench" / "results",
    "realbench": DATA_ROOT / "realbench" / "results",
}

# The label vocabulary shared by the labeling guide, tools, and metrics.
FAILURE_CLASSES = [
    "none", "prompt_drift", "tool_misuse", "context_overflow",
    "hallucination_loop", "goal_misalignment",
]
META_LABELS = ["gap", "multi", "unclear"]
VALID_LABELS = FAILURE_CLASSES + META_LABELS


@dataclass
class Trace:
    """One agent run, normalised across benchmarks."""
    trace_id: str
    benchmark: str
    path: Path
    task: str
    task_class: str
    trace_summary: str
    executor_output: str
    observer_flags: list = field(default_factory=list)
    critic_scores: dict = field(default_factory=dict)
    aria_label: Optional[str] = None          # ARIA's prediction (hidden during labeling)
    expected_answer: str = ""
    answer_correct: Optional[bool] = None     # did the agent get the known-correct answer?
    human_label: Optional[str] = None         # legacy single-annotator label (if present)
    raw: dict = field(default_factory=dict)

    @property
    def aria_label_norm(self) -> str:
        """ARIA prediction as a comparable string ('none' when no failure)."""
        return (self.aria_label or "none").strip().lower()


def _normalise(benchmark: str, path: Path, r: dict) -> Trace:
    """Map a raw result dict to the unified Trace shape."""
    # answer-correct flag has two historical names
    ac = r.get("answer_correct")
    if ac is None:
        ac = r.get("gaia_correct")
    return Trace(
        trace_id=f"{benchmark}/{path.stem}",
        benchmark=benchmark,
        path=path,
        task=r.get("task") or r.get("question") or r.get("task_description", ""),
        task_class=r.get("task_class", "unknown"),
        trace_summary=r.get("trace_summary", ""),
        executor_output=r.get("executor_output", ""),
        observer_flags=r.get("observer_flags") or [],
        critic_scores=r.get("critic_scores") or {},
        aria_label=r.get("aria_label"),
        expected_answer=str(r.get("expected_answer", "")),
        answer_correct=ac,
        human_label=r.get("human_label"),
        raw=r,
    )


def iter_traces(
    benchmarks: Optional[list[str]] = None,
    *,
    include_errors: bool = False,
) -> Iterator[Trace]:
    """Yield every completed trace across the requested benchmarks.

    Errored runs (``run_error`` set) are skipped unless ``include_errors``.
    """
    names = benchmarks or list(RESULT_DIRS)
    for name in names:
        rdir = RESULT_DIRS.get(name)
        if not rdir or not rdir.exists():
            continue
        for fp in sorted(rdir.glob("*.json")):
            try:
                r = json.loads(fp.read_text(encoding="utf-8"))
            except Exception:
                continue
            if r.get("run_error") and not include_errors:
                continue
            yield _normalise(name, fp, r)


def load_traces(
    benchmarks: Optional[list[str]] = None,
    *,
    include_errors: bool = False,
) -> list[Trace]:
    return list(iter_traces(benchmarks, include_errors=include_errors))


def trace_by_id(benchmarks: Optional[list[str]] = None) -> dict[str, Trace]:
    return {t.trace_id: t for t in iter_traces(benchmarks)}
