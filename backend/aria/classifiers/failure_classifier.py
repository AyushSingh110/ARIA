from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

FAILURE_CLASSES = [
    "prompt_drift",
    "tool_misuse",
    "context_overflow",
    "goal_misalignment",
    "hallucination_loop",
    "none",
]


class FailureFeatureExtractor:
    """Extracts a fixed-length numeric feature vector from ARIAState for XGBoost."""

    def extract(self, state: dict) -> np.ndarray:
        flags = state.get("observer_flags", [])
        drift_scores = state.get("drift_scores", [])
        critic = state.get("critic_scores") or {}
        max_turns = 10

        return np.array(
            [
                max(drift_scores, default=0.0),
                float(np.mean(drift_scores)) if drift_scores else 0.0,
                sum(1 for f in flags if f["flag_type"] == "prompt_drift"),
                sum(1 for f in flags if f["flag_type"] == "tool_repetition"),
                sum(1 for f in flags if f["flag_type"] == "tool_error_loop"),
                sum(1 for f in flags if f["flag_type"] == "turn_budget_warning"),
                state.get("executor_turn_count", 0) / max_turns,
                critic.get("correctness", 3.0),
                critic.get("completeness", 3.0),
                critic.get("efficiency", 3.0),
                critic.get("safety", 5.0),
                critic.get("overall", 3.0),
            ],
            dtype=np.float32,
        )


class XGBoostFailureClassifier:
    """XGBoost multiclass classifier over the 5-class failure taxonomy.

    Trained in Phase 5 on 500 benchmark runs. Before training, `is_trained()`
    returns False and `predict()` returns None — Diagnostician falls back to LLM.
    """

    def __init__(self) -> None:
        self._model = None
        self._extractor = FailureFeatureExtractor()

    def is_trained(self) -> bool:
        return self._model is not None

    def load(self, path: str | Path) -> bool:
        try:
            import xgboost as xgb
            model = xgb.XGBClassifier()
            model.load_model(str(path))
            self._model = model
            return True
        except Exception:
            return False

    def predict(self, state: dict) -> Optional[str]:
        if not self._model:
            return None
        features = self._extractor.extract(state).reshape(1, -1)
        idx = int(self._model.predict(features)[0])
        return FAILURE_CLASSES[idx] if idx < len(FAILURE_CLASSES) else None

    def predict_proba(self, state: dict) -> Optional[dict[str, float]]:
        if not self._model:
            return None
        features = self._extractor.extract(state).reshape(1, -1)
        proba = self._model.predict_proba(features)[0]
        return {cls: float(p) for cls, p in zip(FAILURE_CLASSES, proba)}
