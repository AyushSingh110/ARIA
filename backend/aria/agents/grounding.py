from __future__ import annotations

"""Critic v3 — Factual Grounding Verification.

Research motivation (GAIA + RealBench, June 2026):
  Critic v2 checks whether requirements are ADDRESSED, not whether the answer
  is FACTUALLY CORRECT. In 3/4 GAIA false-clean cases and 3 RealBench cases,
  the agent gave a confident wrong answer, Critic v2 scored req_sat=1.0, and
  ARIA returned "none". This module closes that blind spot.

Approach:
  1. Extract the central factual claim from the agent's final answer.
  2. Run an independent web search for the claim.
  3. Ask the LLM: is the claim SUPPORTED / CONTRADICTED / UNVERIFIABLE
     by the search evidence?
  4. CONTRADICTED + high req_sat → hallucination signal.

Enabled via .env:  GROUNDING_ENABLED=true
"""

import json
import re
from typing import Any, Optional, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage

from aria.config import get_settings
from aria.utils.display import console


class GroundingResult(TypedDict):
    verdict: str          # "supported" | "contradicted" | "unverifiable" | "skipped"
    confidence: float     # 0.0–1.0
    claim: str            # the factual claim that was checked
    evidence: str         # snippet of search evidence used
    reasoning: str


_CLAIM_PROMPT = """\
You extract the single most important factual claim from an AI agent's answer.

Rules:
- Return the claim as ONE short declarative sentence containing the key fact
  (a name, number, date, place, or title) and enough context to verify it.
- If the answer contains no verifiable factual claim (e.g. it is a calculation
  shown step by step, a file operation, or an opinion), return exactly: NONE

Respond with ONLY the claim sentence or NONE."""

_VERIFY_PROMPT = """\
You are a fact verification assistant. Given a CLAIM and independent SEARCH
EVIDENCE, decide whether the evidence supports the claim.

Respond with ONLY a valid JSON object:
{
  "verdict": "supported" | "contradicted" | "unverifiable",
  "confidence": <0.0-1.0>,
  "reasoning": "<one sentence>"
}

Rules:
- "supported": evidence clearly confirms the key fact in the claim
- "contradicted": evidence indicates a DIFFERENT answer than the claim
- "unverifiable": evidence is insufficient or off-topic
- Be strict: a claim is only "supported" if the specific fact matches."""


def _get_llm() -> Any:
    from langchain_groq import ChatGroq
    s = get_settings()
    return ChatGroq(api_key=s.groq_api_key, model=s.groq_model, temperature=0)


def _search(query: str, max_results: int = 4) -> str:
    try:
        try:
            from ddgs import DDGS
        except ImportError:
            from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            hits = list(ddgs.text(query, max_results=max_results))
        if not hits:
            return ""
        return "\n".join(
            f"- {h.get('title', '')}: {h.get('body', '')[:200]}"
            for h in hits
        )
    except Exception:
        return ""


def _extract_json(raw: str) -> dict:
    cleaned = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.IGNORECASE)
    cleaned = re.sub(r"```\s*$", "", cleaned.strip())
    return json.loads(cleaned)


def grounding_check(task_description: str, final_answer: str) -> GroundingResult:
    """Verify the agent's final answer against independent web evidence.

    Returns a GroundingResult. verdict="skipped" when there is nothing to
    check (no factual claim, no search evidence, or LLM failure) — callers
    must treat "skipped" and "unverifiable" as non-signals, never as failures.
    """
    skipped: GroundingResult = {
        "verdict": "skipped", "confidence": 0.0,
        "claim": "", "evidence": "", "reasoning": "",
    }

    if not final_answer or len(final_answer.strip()) < 10:
        return skipped

    llm = _get_llm()

    # Step 1 — extract the central factual claim
    try:
        resp = llm.invoke([
            SystemMessage(content=_CLAIM_PROMPT),
            HumanMessage(content=(
                f"Task given to the agent:\n{task_description[:400]}\n\n"
                f"Agent's final answer:\n{final_answer[:800]}"
            )),
        ])
        claim = resp.content.strip()
    except Exception as exc:
        skipped["reasoning"] = f"claim extraction failed: {exc}"
        return skipped

    if not claim or claim.upper().startswith("NONE"):
        skipped["reasoning"] = "no verifiable factual claim in answer"
        return skipped

    # Step 2 — independent web search
    evidence = _search(claim)
    if not evidence:
        return {
            "verdict": "unverifiable", "confidence": 0.0,
            "claim": claim, "evidence": "",
            "reasoning": "no search evidence available",
        }

    # Step 3 — verify
    try:
        resp = llm.invoke([
            SystemMessage(content=_VERIFY_PROMPT),
            HumanMessage(content=f"CLAIM:\n{claim}\n\nSEARCH EVIDENCE:\n{evidence[:1500]}"),
        ])
        parsed = _extract_json(resp.content)
        verdict = str(parsed.get("verdict", "unverifiable")).lower()
        if verdict not in ("supported", "contradicted", "unverifiable"):
            verdict = "unverifiable"
        try:
            confidence = float(parsed.get("confidence", 0.5))
        except (ValueError, TypeError):
            confidence = 0.5
        return {
            "verdict": verdict,
            "confidence": confidence,
            "claim": claim,
            "evidence": evidence[:400],
            "reasoning": str(parsed.get("reasoning", "")),
        }
    except Exception as exc:
        return {
            "verdict": "unverifiable", "confidence": 0.0,
            "claim": claim, "evidence": evidence[:400],
            "reasoning": f"verification failed: {exc}",
        }


def maybe_ground(task_description: str, final_answer: str) -> Optional[GroundingResult]:
    """Run grounding only if enabled in settings. Returns None when disabled."""
    s = get_settings()
    if not getattr(s, "grounding_enabled", False):
        return None
    console.print("[dim]Critic v3: running factual grounding check...[/dim]")
    return grounding_check(task_description, final_answer)
