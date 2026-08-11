"""
tavily_tool.py
--------------
Web search tool using Tavily. Lazy-initialised so that a missing
TAVILY_API_KEY does not crash the server at startup — the tool returns
a helpful error message when invoked without a key.
"""

from langchain.tools import tool

_tavily_client = None


def _get_tavily():
    """Return a cached TavilySearch instance, or None if key is missing."""
    global _tavily_client
    if _tavily_client is not None:
        return _tavily_client
    try:
        from langchain_tavily import TavilySearch
        _tavily_client = TavilySearch(
            max_results=3,
            search_depth="advanced",
        )
        return _tavily_client
    except Exception:
        return None


@tool
async def web_search_tool(query: str) -> str:
    """
    Search the web for CURRENT information not available in the syllabus
    database — exam patterns, placement trends, curriculum updates, or
    general ECE career questions.

    Do NOT use this for questions about specific course content, unit topics,
    credits, or CGPA — those have dedicated tools.

    Args:
        query: The search query to look up on the web.

    Returns:
        A summary of the top web search results.
    """
    client = _get_tavily()
    if client is None:
        return (
            "Web search is not available. Please set the TAVILY_API_KEY "
            "environment variable to enable this feature."
        )
    try:
        results = await client.ainvoke({"query": query})
        if isinstance(results, list):
            parts = []
            for r in results:
                title   = r.get("title", "")
                url     = r.get("url", "")
                content = r.get("content", "")
                parts.append(f"**{title}**\n{url}\n{content}")
            return "\n\n".join(parts) if parts else "No results found."
        return str(results)
    except Exception as exc:
        return f"Web search failed: {exc}"