from .cgpa_tool import calculate_cgpa
from .tavily_tool import web_search_tool
from .rag_tool import syllabus_rag_search, syllabus_rag_search_async

# all_tools is the list passed to llm.bind_tools() in ChatService.
# Order matters for the system prompt — most specific tools first.
all_tools = [calculate_cgpa, syllabus_rag_search, web_search_tool]