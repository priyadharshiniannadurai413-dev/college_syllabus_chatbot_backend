"""
Local tool registry.

MCP (GitHub) tools are NOT merged here anymore — they are per-user and
resolved at request time from MCPClientManager (see app/services/mcp_client.py).
ChatService merges local tools with the requesting user's MCP tools when
building the tool map for a conversation.
"""

from typing import Dict, List

from langchain_core.tools import BaseTool

from .cgpa_tool import calculate_cgpa
from .tavily_tool import web_search_tool
from .rag_tool import syllabus_rag_search, syllabus_rag_search_async

all_tools: List[BaseTool] = [calculate_cgpa, syllabus_rag_search, web_search_tool]

# tool_registry maps tool name -> tool instance for the router's subset selection.
tool_registry: Dict[str, BaseTool] = {
    "calculate_cgpa": calculate_cgpa,
    "syllabus_rag_search": syllabus_rag_search,
    "web_search_tool": web_search_tool,
}


def get_all_tools() -> List[BaseTool]:
    """Return the local tools."""
    return list(all_tools)


def get_tool_registry() -> Dict[str, BaseTool]:
    """Return the local tool registry keyed by name."""
    return dict(tool_registry)


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
