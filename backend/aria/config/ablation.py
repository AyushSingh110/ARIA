"""Ablation configuration (Step 2.2).

Lets the same test traces be run through progressively richer configurations so
each component's contribution can be measured — the controlled experiment that
turns the "evolution story" into a defensible result table:

    Config  Components                                   Expected
    A       holistic critic + zero-shot LLM              ~baseline
    B       + Critic v2 (requirement-aware)              big jump
    C       + deterministic disambiguation rules         moderate
    D       + DSPy v2 compiled + Critic v3 grounding     best (full system)

Select a preset with the env var ``ARIA_ABLATION=a|b|c|d`` (case-insensitive).
With no env var set, the FULL system (preset D) runs — so production and normal
development are completely unaffected by this module.

Individual components can also be toggled directly (useful for one-off
component ablations like "does XGBoost help?", Step 3.4):
    ARIA_ABL_CRITIC_V2, ARIA_ABL_RULES, ARIA_ABL_DSPY,
    ARIA_ABL_GROUNDING, ARIA_ABL_XGB   = on/off (override the preset).

The eval harness sets these env vars per run, so after changing them in-process
call ``get_ablation.cache_clear()`` to pick up the new values.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache


@dataclass(frozen=True)
class AblationConfig:
    """Which diagnosis components are active for this run."""
    name: str
    critic_v2: bool       # requirement-aware Critic (False → holistic v1-style critic)
    rules: bool           # deterministic disambiguation rules in the Diagnostician
    dspy_compiled: bool   # load the compiled v2 DSPy program (False → zero-shot)
    grounding: bool       # Critic v3 factual grounding (hallucination catch)
    xgboost: bool         # XGBoost fallback prediction


# Cumulative presets A→D. D is the full production system.
_PRESETS: dict[str, AblationConfig] = {
    "a": AblationConfig("A (holistic + zero-shot)", False, False, False, False, False),
    "b": AblationConfig("B (+Critic v2)",            True,  False, False, False, False),
    "c": AblationConfig("C (+rules)",                True,  True,  False, False, False),
    "d": AblationConfig("D (full: DSPy v2 + grounding)", True, True, True, True, True),
}
_FULL = _PRESETS["d"]


def _override(env_name: str, default: bool) -> bool:
    """Read an on/off env override; fall back to the preset value."""
    v = os.environ.get(env_name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "on", "yes", "y")


@lru_cache(maxsize=1)
def get_ablation() -> AblationConfig:
    """Resolve the active ablation config from the environment.

    Default (no ARIA_ABLATION set) = full system, so normal runs are unchanged.
    """
    preset_key = os.environ.get("ARIA_ABLATION", "").strip().lower()
    base = _PRESETS.get(preset_key, _FULL)
    name = base.name if preset_key in _PRESETS else "full (default)"
    return AblationConfig(
        name=name,
        critic_v2=_override("ARIA_ABL_CRITIC_V2", base.critic_v2),
        rules=_override("ARIA_ABL_RULES", base.rules),
        dspy_compiled=_override("ARIA_ABL_DSPY", base.dspy_compiled),
        grounding=_override("ARIA_ABL_GROUNDING", base.grounding),
        xgboost=_override("ARIA_ABL_XGB", base.xgboost),
    )
