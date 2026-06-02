from __future__ import annotations

from typing import Literal

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from aria.agents import executor_node, observer_node, orchestrator_node
from aria.state.schema import ARIAState


# ── Routing logic ─────────────────────────────────────────────────────────────

def _route_after_observer(
    state: ARIAState,
) -> Literal["orchestrator", "__end__"]:
    """Decide what happens after Observer finishes.

    - more subtasks → back to orchestrator (cyclic)
    - escalated / complete → END
    """
    if state.get("escalate"):
        return "__end__"
    if state.get("current_phase") == "complete":
        return "__end__"
    # More subtasks pending: loop back
    return "orchestrator"


# ── Graph builder ─────────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    """Build the Phase 1 ARIA graph (in-memory, no checkpointing)."""
    graph = StateGraph(ARIAState)

    # Register nodes
    graph.add_node("orchestrator", orchestrator_node)
    graph.add_node("executor", executor_node)
    graph.add_node("observer", observer_node)

    # Fixed edges
    graph.add_edge(START, "orchestrator")
    graph.add_edge("orchestrator", "executor")
    graph.add_edge("executor", "observer")

    # Cyclic conditional edge: observer → orchestrator (loop) or END
    graph.add_conditional_edges(
        "observer",
        _route_after_observer,
        {
            "orchestrator": "orchestrator",
            "__end__": END,
        },
    )

    return graph.compile()


def build_graph_with_checkpointer() -> StateGraph:
    """Build the ARIA graph with in-memory checkpointing for run resumption."""
    graph = StateGraph(ARIAState)

    graph.add_node("orchestrator", orchestrator_node)
    graph.add_node("executor", executor_node)
    graph.add_node("observer", observer_node)

    graph.add_edge(START, "orchestrator")
    graph.add_edge("orchestrator", "executor")
    graph.add_edge("executor", "observer")
    graph.add_conditional_edges(
        "observer",
        _route_after_observer,
        {
            "orchestrator": "orchestrator",
            "__end__": END,
        },
    )

    checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer)
