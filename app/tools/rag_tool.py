"""
rag_tool.py
-----------
Exposes the existing MongoDB Atlas vector-search retrieval pipeline as a
LangChain ``@tool`` so the agent can route syllabus / course questions here.

The tool is *async-safe*: it calls ``VectorStore.retrieve()`` directly and
returns a plain text string the LLM can read as a ToolMessage.
"""

import asyncio
import logging

from langchain.tools import tool

from app.rag.query_normalizer import normalize_query
from app.rag.vector_store import get_vector_store

logger = logging.getLogger("uvicorn")


def _run_retrieval(query: str) -> str:
    """
    Synchronous wrapper around the async VectorStore.retrieve().
    Used so the @tool decorator (which expects a sync function) can
    still drive the async retrieve call.
    """
    try:
        # Check if an event loop is currently running in this thread
        try:
            loop = asyncio.get_running_loop()
            in_loop = True
        except RuntimeError:
            in_loop = False

        if in_loop:
            # Inside an active async context — use a new thread-safe loop
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, _async_retrieve(query))
                return future.result()
        else:
            # No active loop in this thread (e.g. LangChain's ainvoke thread pool)
            return asyncio.run(_async_retrieve(query))
    except Exception as exc:
        logger.error(f"[syllabus_rag_search] Retrieval failed: {exc}")
        return f"Retrieval failed: {exc}"


async def _async_retrieve(query: str) -> str:
    """Core async retrieval logic shared by both the sync wrapper and direct async calls."""
    normalized = normalize_query(query)
    search_query = normalized["search_query"]
    where_filter = normalized["where"]

    vector_store = get_vector_store()
    results = await vector_store.retrieve(search_query, top_k=4, where=where_filter)

    matches = results.get("matches", [])
    if not matches:
        return "No relevant syllabus information found for your query."

    chunks = []
    for i, match in enumerate(matches, start=1):
        text = match.get("text", "").strip()
        meta = match.get("metadata", {})
        semester = meta.get("semester", "")
        source = match.get("source", "")
        header = f"[Chunk {i}" + (f" | Semester {semester}" if semester else "") + (f" | {source}" if source else "") + "]"
        chunks.append(f"{header}\n{text}")

    return "\n\n".join(chunks)


@tool
async def syllabus_rag_search(query: str) -> str:
    """
    Search the college syllabus database for course content, subject lists,
    unit topics, credit information, course objectives, course outcomes,
    and fee structure.

    Use this tool for ANY question about:
    - Specific subjects or courses (e.g., 'What is taught in Data Structures?')
    - Semester-wise subject lists (e.g., 'What courses are in semester 3?')
    - Unit-level syllabus content (e.g., 'What are the topics in Unit 2 of OS?')
    - Credit structure and total credits
    - Course objectives or outcomes
    - Fee structure

    Do NOT use this for CGPA calculations or current/live web information.

    Args:
        query: The syllabus-related question to search for.

    Returns:
        Relevant syllabus content retrieved from the vector database.
    """
    return await _async_retrieve(query)


async def syllabus_rag_search_async(query: str) -> str:
    """
    Async version of syllabus_rag_search for use inside async tool-execution
    loops in ChatService. Bypasses the sync wrapper entirely.
    """
    return await _async_retrieve(query)
