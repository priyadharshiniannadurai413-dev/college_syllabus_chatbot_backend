from typing import Dict, List
from langchain_core.tools import BaseTool

from .cgpa_tool import calculate_cgpa
from .tavily_tool import web_search_tool
from .rag_tool import syllabus_rag_search, syllabus_rag_search_async

# Local tools only — used for module-level references if needed.
all_tools: List[BaseTool] = [calculate_cgpa, syllabus_rag_search, web_search_tool]

# tool_registry maps tool name -> tool instance for the router's subset selection.
tool_registry: Dict[str, BaseTool] = {
    "calculate_cgpa": calculate_cgpa,
    "syllabus_rag_search": syllabus_rag_search,
    "web_search_tool": web_search_tool,
}

# Cached merged registries — invalidated when MCP tools change
_cached_all_tools: List[BaseTool] | None = None
_cached_tool_registry: Dict[str, BaseTool] | None = None


def invalidate_tool_cache():
    """Call after MCP tools change to rebuild cached registries."""
    global _cached_all_tools, _cached_tool_registry
    _cached_all_tools = None
    _cached_tool_registry = None
    # Also invalidate chat service's tool map
    try:
        from app.ai import chat_service
        chat_service._cached_tool_map = None
    except ImportError:
        pass


def get_all_tools() -> List[BaseTool]:
    """Return local tools merged with MCP tools (GitHub, etc.). Cached after first call."""
    global _cached_all_tools
    if _cached_all_tools is not None:
        return _cached_all_tools
    from app.services.mcp_client import mcp_client
    _cached_all_tools = all_tools + mcp_client.get_tools()
    return _cached_all_tools


def get_tool_registry() -> Dict[str, BaseTool]:
    """Return merged registry of local + MCP tools keyed by name. Cached after first call."""
    global _cached_tool_registry
    if _cached_tool_registry is not None:
        return _cached_tool_registry
    from app.services.mcp_client import mcp_client
    registry = dict(tool_registry)
    for t in mcp_client.get_tools():
        registry[t.name] = t
    _cached_tool_registry = registry
    return _cached_tool_registry


def get_mcp_tool_registry() -> Dict[str, BaseTool]:
    """Return only MCP tools keyed by name — used for GitHub intent routing."""
    from app.services.mcp_client import mcp_client
    return {t.name: t for t in mcp_client.get_tools()}


# tool_categories — metadata for the hybrid router.
# Each tool maps to a list of intent tags the router uses for classification
# and for the LLM-based fallback router prompt.
tool_categories = {
    "calculate_cgpa": ["cgpa", "gpa", "sgpa", "grade", "calculation", "academic_score"],
    "syllabus_rag_search": [
        "syllabus", "subject", "course", "unit", "module", "credits",
        "semester", "objectives", "outcomes", "topics", "curriculum",
        "regulation", "cbc", "department", "lab", "practical", "theory",
    ],
    "web_search_tool": [
        "latest", "current", "trend", "placement", "news", "exam",
        "industry", "career", "job", "recruitment", "gate", "interview",
        "company", "salary", "package",
    ],
    "github": [
        "github", "repo", "repository", "issue", "pull_request", "pr",
        "commit", "branch", "merge", "fork", "code", "clone", "push",
    ],
}
