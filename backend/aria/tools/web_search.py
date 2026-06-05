from __future__ import annotations

from langchain_core.tools import tool

# Real DuckDuckGo search — no API key required.
# Falls back to a minimal stub only if duckduckgo-search is not installed.


def _fetch_results(query: str, max_results: int = 5) -> str:
    try:
        try:
            from ddgs import DDGS
        except ImportError:
            from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            hits = list(ddgs.text(query, max_results=max_results))
        if not hits:
            return f"No results found for: {query}"
        lines = []
        for i, h in enumerate(hits, 1):
            title = h.get("title", "")
            body  = h.get("body", "")
            href  = h.get("href", "")
            lines.append(f"{i}. {title}\n   {body[:200]}\n   Source: {href}")
        return "\n\n".join(lines)
    except ImportError:
        return (
            f"[web_search stub] Query: {query}\n"
            "Install real search: pip install duckduckgo-search"
        )
    except Exception as exc:
        return f"Search error: {exc}"


@tool
def web_search(query: str) -> str:
    """Search the web for information about a topic.

    Args:
        query: The search query string.

    Returns:
        Top search results as plain text with titles, snippets, and sources.
    """
    if not query.strip():
        return "Error: query cannot be empty."
    return _fetch_results(query.strip())
