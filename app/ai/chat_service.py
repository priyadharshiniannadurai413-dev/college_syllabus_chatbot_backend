"""
chat_service.py
---------------
Orchestrates the tool-calling agent pipeline:

    SystemPrompt + bind_tools(all_tools)
        → LLM (Groq primary / Gemini fallback)
        → tool_calls? → execute tool → ToolMessage
        → LLM final answer (streamed token-by-token)

Tools registered:
    calculate_cgpa        — CGPA / semester GPA math
    syllabus_rag_search   — Course content, units, credits, objectives
    web_search_tool       — Exam patterns, placements, current live info
"""

import json
import logging
from typing import AsyncIterator

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI

from app.core.config import settings
from app.tools import all_tools

logger = logging.getLogger("uvicorn")

# ──────────────────────────────────────────────────────────────────────────────
# System prompt — instructs the LLM when to call each tool
# NOTE: Do NOT describe the JSON tool-calling format here. Native tool calling
# is handled at the API level via bind_tools() — describing the format in the
# prompt text can cause the model to type it out as plain text instead of
# actually invoking the tool.
# ──────────────────────────────────────────────────────────────────────────────
_SYSTEM_PROMPT = """You are a helpful college assistant for an ECE (Electronics and Communication Engineering) department.

You have access to three tools. Always prefer a tool over guessing:

1. calculate_cgpa
   Use when the user asks to calculate CGPA or cumulative GPA from semester GPA values.
   Example triggers: "what is my CGPA", "calculate my GPA", "my semester GPAs are 8.0, 7.5, 9.0"

2. syllabus_rag_search
   Use when the user asks about course content, subjects, units, topics, credits,
   objectives, outcomes, or fee structure from the college syllabus.
   Example triggers: "what subjects are in semester 3", "explain unit 2 of data structures",
   "how many credits does OS have", "what are the course objectives of CN"

3. web_search_tool
   Use when the user asks for CURRENT information not available in the syllabus database —
   exam patterns, placement statistics, industry trends, or general ECE career questions.
   Example triggers: "latest placement trends for ECE", "GATE 2025 exam pattern",
   "career opportunities in VLSI"

If the question is a simple greeting or does not require any tool, answer directly.
Always respond in clear, concise English. Format lists and tables when helpful.
"""

# ──────────────────────────────────────────────────────────────────────────────
# Tool executor — maps tool name → async call
# ──────────────────────────────────────────────────────────────────────────────
_TOOL_MAP = {t.name: t for t in all_tools}

async def _execute_tool(tool_name: str, tool_args: dict) -> str:
    """
    Execute a tool by name dynamically using the registered _TOOL_MAP.
    Utilizes LangChain's native .ainvoke() which safely handles both
    synchronous and asynchronous tools without blocking the event loop.
    """
    logger.info(f"[ChatService] Executing tool '{tool_name}' with args: {tool_args}")

    try:
        tool = _TOOL_MAP.get(tool_name)
        if not tool:
            logger.warning(f"[ChatService] Unknown tool requested: {tool_name}")
            return f"Unknown tool: {tool_name}"

        result = await tool.ainvoke(tool_args)
        return str(result)

    except Exception as exc:
        logger.error(f"[ChatService] Tool execution failed for '{tool_name}': {exc}", exc_info=True)
        return f"Tool execution error: {exc}"


class ChatService:
    """
    Orchestrates the tool-calling agent pipeline using native LangChain LLM wrappers
    and native .with_fallbacks().

    Flow:
        1. Build [SystemMessage, HumanMessage]
        2. primary_llm.with_fallbacks([fallback_llm])
        3. bind_tools() and invoke()
        4. If tool_calls present: execute tools, append ToolMessages
        5. stream final response natively.
    """

    def __init__(self, user_prompt: str, is_voice: bool = False):
        self.user_prompt = user_prompt
        self.is_voice = is_voice

        # ── Initialize LLMs ───────────────────────────────────────────────────
        self.primary_llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.7,
            max_tokens=1024,
            api_key=settings.GROQ_API_KEY,
        )

        self.fallback_llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0.7,
            max_tokens=1024,
            api_key=settings.GEMINI_API_KEY,
        )

    async def chat(self) -> AsyncIterator[str]:
        """
        Async generator — yields response text chunks token-by-token for
        StreamingResponse in the FastAPI route.
        """
        messages = [
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=self.user_prompt),
        ]

        parser = StrOutputParser()

        # Build tool-bound models — tool_choice="auto" makes the model decide
        # whether to call a tool or answer directly, using NATIVE function
        # calling (not text-based imitation).
        primary_with_tools = self.primary_llm.bind_tools(all_tools, tool_choice="auto")
        fallback_with_tools = self.fallback_llm.bind_tools(all_tools, tool_choice="auto")

        # Create combined fallback chain for the first round (tool inspection)
        llm_with_tools_chain = primary_with_tools.with_fallbacks([fallback_with_tools])

        try:
            logger.info("[ChatService] Round 1: Invoking LLM with tool definitions.")
            first_response: AIMessage = await llm_with_tools_chain.ainvoke(messages)

            tool_calls = getattr(first_response, "tool_calls", None)

            # Defensive check: sometimes a model with weak tool-calling support
            # will emit a tool-call-looking JSON string as plain content instead
            # of using the structured tool_calls field. Catch that case explicitly
            # so it doesn't leak to the user as raw JSON.
            raw_content = (first_response.content or "").strip()
            looks_like_fake_tool_call = raw_content.startswith('{"type": "function"') or (
                raw_content.startswith("{") and '"name"' in raw_content and '"parameters"' in raw_content
            )

            if not tool_calls and looks_like_fake_tool_call:
                logger.warning(
                    "[ChatService] Model emitted a fake text tool-call instead of "
                    "using native tool_calls. Retrying without tools on fallback LLM."
                )
                # Re-ask without tool binding to force a plain-text answer,
                # or you could re-route straight to syllabus_rag_search here
                # if you want a guaranteed fallback behavior instead.
                plain_chain = self.fallback_llm | parser
                async for chunk in plain_chain.astream(messages):
                    if chunk:
                        yield chunk
                return

            # ── No tool call — stream the answer directly ─────────────────────
            if not tool_calls:
                logger.info("[ChatService] Direct answer (no tool call)")
                if raw_content:
                    yield raw_content
                else:
                    stream_chain = self.primary_llm.with_fallbacks([self.fallback_llm]) | parser
                    async for chunk in stream_chain.astream(messages):
                        if chunk:
                            yield chunk
                return

            # ── Tool call(s) detected — execute each tool ─────────────────────
            logger.info(f"[ChatService] Tool calls requested: {[tc.get('name') for tc in tool_calls]}")
            messages.append(first_response)

            for tc in tool_calls:
                tc_id = tc.get("id", "")
                tc_name = tc.get("name", "")
                tc_args = tc.get("args", {})

                if isinstance(tc_args, str):
                    try:
                        tc_args = json.loads(tc_args)
                    except json.JSONDecodeError:
                        tc_args = {}

                tool_output = await _execute_tool(tc_name, tc_args)
                logger.info(f"[ChatService] Tool '{tc_name}' output (first 200 chars): {tool_output[:200]}")

                messages.append(
                    ToolMessage(
                        content=tool_output,
                        tool_call_id=tc_id,
                    )
                )

            # ── Round 2: Stream the final answer ──────────────────────────────
            logger.info("[ChatService] Streaming final answer after tool use.")
            stream_chain = llm_with_tools_chain | parser
            async for chunk in stream_chain.astream(messages):
                if chunk:
                    yield chunk

        except Exception as exc:
            logger.error(f"[ChatService] Chat failed entirely: {exc}")
            yield "I'm currently experiencing technical difficulties. Please try again later."