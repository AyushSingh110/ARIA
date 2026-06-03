from __future__ import annotations

from typing import Literal

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from aria.agents import (
    critic_node,
    diagnostician_node,
    executor_node,
    observer_node,
    orchestrator_node,
    refiner_node,
    validator_node,
)
from aria.state.schema import ARIAState


# ── Routing helpers ───────────────────────────────────────────────────────────

def _route_after_observer_p1(state: ARIAState) -> Literal["orchestrator", "__end__"]:
    if state.get("escalate") or state.get("current_phase") == "complete":
        return "__end__"
    return "orchestrator"


def _route_after_diagnostician(state: ARIAState) -> str:
    if state.get("escalate") or state.get("current_phase") == "escalated":
        return "__end__"

    failure = state.get("failure_class")
    pass_fail = (state.get("critic_scores") or {}).get("pass_fail", True)

    # Failure detected and run didn't pass → send to Refiner
    if failure and not pass_fail:
        return "refiner"

    # Clean run — more subtasks or done
    if state.get("current_phase") == "complete":
        return "__end__"
    return "orchestrator"


def _route_after_validator(state: ARIAState) -> str:
    if state.get("escalate") or state.get("current_phase") == "escalated":
        return "__end__"
    if state.get("committed_to_store") or state.get("current_phase") == "complete":
        return "__end__"
    return "refiner"    # retry with a different refinement


# ── Phase graph builders ──────────────────────────────────────────────────────

def build_graph_p1():
    """Phase 1: orchestrator → executor → observer → (loop | END)."""
    g = StateGraph(ARIAState)
    g.add_node("orchestrator", orchestrator_node)
    g.add_node("executor", executor_node)
    g.add_node("observer", observer_node)
    g.add_edge(START, "orchestrator")
    g.add_edge("orchestrator", "executor")
    g.add_edge("executor", "observer")
    g.add_conditional_edges(
        "observer",
        _route_after_observer_p1,
        {"orchestrator": "orchestrator", "__end__": END},
    )
    return g.compile()


def build_graph_p2():
    """Phase 2: adds Critic + Diagnostician."""
    g = StateGraph(ARIAState)
    g.add_node("orchestrator", orchestrator_node)
    g.add_node("executor", executor_node)
    g.add_node("observer", observer_node)
    g.add_node("critic", critic_node)
    g.add_node("diagnostician", diagnostician_node)
    g.add_edge(START, "orchestrator")
    g.add_edge("orchestrator", "executor")
    g.add_edge("executor", "observer")
    g.add_edge("observer", "critic")
    g.add_edge("critic", "diagnostician")
    g.add_conditional_edges(
        "diagnostician",
        _route_after_diagnostician,
        {"orchestrator": "orchestrator", "refiner": "__end__", "__end__": END},
    )
    return g.compile()


def build_graph():
    """Phase 3 (current): full closed loop with Refiner + Validator."""
    g = StateGraph(ARIAState)
    g.add_node("orchestrator", orchestrator_node)
    g.add_node("executor", executor_node)
    g.add_node("observer", observer_node)
    g.add_node("critic", critic_node)
    g.add_node("diagnostician", diagnostician_node)
    g.add_node("refiner", refiner_node)
    g.add_node("validator", validator_node)

    g.add_edge(START, "orchestrator")
    g.add_edge("orchestrator", "executor")
    g.add_edge("executor", "observer")
    g.add_edge("observer", "critic")
    g.add_edge("critic", "diagnostician")

    g.add_conditional_edges(
        "diagnostician",
        _route_after_diagnostician,
        {"orchestrator": "orchestrator", "refiner": "refiner", "__end__": END},
    )

    g.add_edge("refiner", "validator")

    g.add_conditional_edges(
        "validator",
        _route_after_validator,
        {"refiner": "refiner", "__end__": END},
    )

    return g.compile()


def build_graph_with_checkpointer():
    """Phase 3 graph with in-memory checkpointing."""
    g = StateGraph(ARIAState)
    g.add_node("orchestrator", orchestrator_node)
    g.add_node("executor", executor_node)
    g.add_node("observer", observer_node)
    g.add_node("critic", critic_node)
    g.add_node("diagnostician", diagnostician_node)
    g.add_node("refiner", refiner_node)
    g.add_node("validator", validator_node)

    g.add_edge(START, "orchestrator")
    g.add_edge("orchestrator", "executor")
    g.add_edge("executor", "observer")
    g.add_edge("observer", "critic")
    g.add_edge("critic", "diagnostician")
    g.add_conditional_edges(
        "diagnostician",
        _route_after_diagnostician,
        {"orchestrator": "orchestrator", "refiner": "refiner", "__end__": END},
    )
    g.add_edge("refiner", "validator")
    g.add_conditional_edges(
        "validator",
        _route_after_validator,
        {"refiner": "refiner", "__end__": END},
    )
    return g.compile(checkpointer=MemorySaver())
