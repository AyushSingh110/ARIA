from __future__ import annotations

from langchain_core.tools import tool

# Phase 1: Mock web search that returns plausible canned results.
# Phase 2+: swap the body of _fetch_results for a real API call
# (SerpAPI / Tavily / DuckDuckGo) without changing any other code.

_MOCK_RESULTS: dict[str, str] = {
    "default": (
        "Search results for '{query}':\n"
        "1. Overview article — general information about {query}.\n"
        "2. Wikipedia entry — background and history of {query}.\n"
        "3. Recent news — latest developments related to {query}.\n"
        "[Phase 1 mock — real search not yet wired]"
    ),
    "python": (
        "Python 3.12 released October 2023. Key features: improved error messages, "
        "faster CPython interpreter (~5 % speedup), new type parameter syntax. "
        "Docs: docs.python.org"
    ),
    "llm": (
        "Large Language Models (LLMs) are transformer-based neural networks trained on "
        "large text corpora. Leading models include GPT-4, Claude 3, Llama 3.1. "
        "Key papers: Attention Is All You Need (2017), GPT-3 (2020), LLaMA (2023)."
    ),
    "langgraph": (
        "LangGraph is an orchestration library for cyclic multi-agent LLM applications. "
        "Built on LangChain. Supports StateGraph with typed state, conditional edges, "
        "checkpointing. Version 0.2 adds first-class multi-agent support."
    ),
}


def _fetch_results(query: str) -> str:
    query_lower = query.lower()
    for keyword, result in _MOCK_RESULTS.items():
        if keyword != "default" and keyword in query_lower:
            return result
    return _MOCK_RESULTS["default"].format(query=query)


@tool
def web_search(query: str) -> str:
    """Search the web for information about a topic.

    Args:
        query: The search query string.

    Returns:
        Summarised search results as plain text.
    """
    if not query.strip():
        return "Error: query cannot be empty."
    return _fetch_results(query.strip())
