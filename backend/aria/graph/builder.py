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
)
from aria.state.schema import ARIAState


def _route_after_observer_p1(state: ARIAState) -> Literal["orchestrator", "__end__"]:
    if state.get("escalate") or state.get("current_phase") == "complete":
        return "__end__"
    return "orchestrator"


def _route_after_diagnostician(state: ARIAState) -> Literal["orchestrator", "__end__"]:
    if state.get("escalate") or state.get("current_phase") == "complete":
        return "__end__"
    return "orchestrator"


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


def build_graph():
    """Phase 2: orchestrator → executor → observer → critic → diagnostician → (loop | END)."""
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
        {"orchestrator": "orchestrator", "__end__": END},
    )
    return g.compile()


def build_graph_with_checkpointer():
    """Phase 2 graph with in-memory checkpointing."""
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
        {"orchestrator": "orchestrator", "__end__": END},
    )
    return g.compile(checkpointer=MemorySaver())
