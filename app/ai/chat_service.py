"""
chat_service.py
---------------
Orchestrates the tool-calling agent pipeline with hybrid router-based tool selection:

    User Query
        ↓
    Layer 1: Regex Router (zero-cost, fast)
        → GREETING → short-circuit, direct response
        → High-confidence → route directly to tool subset
        → Low-confidence → Layer 2
            ↓
    Layer 2: LLM Router (Gemini Flash, only for ambiguous queries)
        → Classifies intent + selects tools with conversation context
        ↓
    SystemPrompt + bind_tools(tool_subset)
        → LLM (Mistral primary / Gemini fallback)
        → tool_calls? → execute tool(s) → ToolMessage(s)
        → LLM final answer (streamed token-by-token)

Tools registered:
    calculate_cgpa        — CGPA / semester GPA math
    syllabus_rag_search   — Course content, units, credits, objectives
    web_search_tool       — Exam patterns, placements, current live info
    GitHub MCP tools      — per-user, only when the requesting user has
                            connected their own GitHub account (OAuth)
"""

import json
import asyncio
import logging
from typing import AsyncIterator, List, Optional

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import BaseTool
from langchain_mistralai import ChatMistralAI
from langchain_google_genai import ChatGoogleGenerativeAI

from app.core.config import settings
from app.tools import get_tool_registry
from app.ai.router import route, route_with_llm, Intent
from app.services.mcp_client import get_connected_client_for_user, MCPClient

logger = logging.getLogger("uvicorn")

# Cached tool map for _execute_tool — avoids rebuilding dict per call
_cached_tool_map: dict[str, BaseTool] | None = None

# ──────────────────────────────────────────────────────────────────────────────
# System prompt — instructs the LLM when to call each tool
# NOTE: Do NOT describe the JSON tool-calling format here. Native tool calling
# is handled at the API level via bind_tools() — describing the format in the
# prompt text can cause the model to type it out as plain text instead of
# actually invoking the tool.
# ──────────────────────────────────────────────────────────────────────────────
_SYSTEM_PROMPT = """You are an ECE department college assistant. Use tools, don't guess.

Tools:
- calculate_cgpa: CGPA/GPA calculations from semester GPAs
- syllabus_rag_search: Course content, subjects, units, credits, objectives, syllabus queries
- web_search_tool: Current info — placements, exam patterns, career trends, industry news
- GitHub tools: Repo search, issues, PRs, commits, branches — any GitHub operation

## GitHub repo resolution — MANDATORY workflow
When the user mentions a GitHub repo by name or description but NOT as an explicit
owner/repo link (e.g. "readme of my Instagram-clone project"), you do NOT know the
owner yet. Resolve it first:
1. Call get_me → learn the authenticated username (this is the owner).
2. Call search_repositories with keywords from the user's phrase
   (e.g. query "instagram clone" or "user:<username> instagram clone in:name").
3a. Exactly one clear match → use its full_name ("owner/repo") for the actual
    request (get_file_contents, list_issues, etc.) and answer.
3b. Multiple matches → list the top matches (name + short description) and ask
    the user which one they meant.
3c. No match → tell the user no such repo was found in their account.
NEVER invent owner or repo names. NEVER call get_file_contents / issue / PR /
commit tools with a guessed owner. If the user gave a full URL like
github.com/owner/repo, extract owner and repo directly and skip steps 1-2.

If the question is a simple greeting or needs no tool, answer directly. Be concise. Use tables/lists for structured data."""

# ──────────────────────────────────────────────────────────────────────────────
# Tool executor — per-request tool map (local tools + the requesting user's
# MCP tools). Built in ChatService.chat() so GitHub calls run as THAT user.
# ──────────────────────────────────────────────────────────────────────────────


# ──────────────────────────────────────────────────────────────────────────────
# ChatService
# ──────────────────────────────────────────────────────────────────────────────
class ChatService:
    """
    Orchestrates the tool-calling agent pipeline using native LangChain LLM wrappers,
    native .with_fallbacks(), and hybrid router-based tool subset selection.

    Flow:
        1. Regex router classifies intent → high-confidence route or low-confidence flag
        2. If low-confidence and router_llm available → LLM router refines decision
        3. Build [SystemMessage, HumanMessage]
        4. primary_llm.bind_tools(tool_subset)
        5. invoke()
        6. If tool_calls present: execute tools concurrently → ToolMessages
        7. stream final response natively.
    """

    def __init__(
        self,
        user_prompt: str,
        is_voice: bool = False,
        conversation_history: Optional[List[dict]] = None,
        user_id: Optional[str] = None,
    ):
        self.user_prompt = user_prompt
        self.is_voice = is_voice
        self.conversation_history = conversation_history
        self.user_id = user_id  # Clerk user id — resolves per-user GitHub MCP tools
        # Per-request tool map (local tools + this user's MCP tools), built in chat()
        self.tool_map: dict[str, BaseTool] = {}

        # ── Initialize LLMs ───────────────────────────────────────────────────
        self.primary_llm = ChatMistralAI(
            model="mistral-small-latest",
            temperature=0.7,
            max_tokens=1024,
            api_key=settings.MISTRAL_API_KEY,
        )

        # NOTE: gemini-flash-latest is a version-less alias — it always resolves
        # to Google's current flash model, so it never 404s when versions retire
        # (gemini-1.5-flash was retired and broke this fallback silently).
        self.fallback_llm = ChatGoogleGenerativeAI(
            model="gemini-flash-latest",
            temperature=0.7,
            max_tokens=1024,
            api_key=settings.GEMINI_API_KEY,
        )

        # Lightweight LLM for the router (Layer 2) — fast, cheap
        self.router_llm = ChatGoogleGenerativeAI(
            model="gemini-3.6-flash",
            temperature=0,
            max_tokens=200,
            api_key=settings.GEMINI_API_KEY,
        )

    # ── Tool execution ────────────────────────────────────────────────────

    @staticmethod
    def _is_auth_failure(message: str) -> bool:
        lowered = message.lower()
        return (
            "401" in lowered
            or "bad credentials" in lowered
            or "unauthorized" in lowered
        )

    async def _invalidate_github_access(self) -> None:
        """
        The user's GitHub token was rejected (revoked/expired) — drop the MCP
        client and the stored token so the next attempt prompts a reconnect.
        """
        if not self.user_id:
            return
        logger.warning(
            f"[ChatService] GitHub auth failure for user {self.user_id} — "
            f"invalidating stored token"
        )
        from app.services.mcp_client import mcp_manager
        from app.db.token_store import delete_token

        mcp_manager.disconnect_user(self.user_id)
        try:
            await delete_token(self.user_id)
        except Exception as exc:
            logger.error(f"[ChatService] Failed to delete revoked token: {exc}")

    async def _execute_tool(self, tool_name: str, tool_args: dict) -> str:
        """
        Execute a tool by name using this request's tool map.
        Utilizes LangChain's native .ainvoke() which safely handles both
        synchronous and asynchronous tools without blocking the event loop.
        """
        logger.info(f"[ChatService] Executing tool '{tool_name}' with args: {tool_args}")

        try:
            tool = self.tool_map.get(tool_name)
            if not tool:
                logger.warning(f"[ChatService] Unknown tool requested: {tool_name}")
                return f"Unknown tool: {tool_name}"

            result = await tool.ainvoke(tool_args)
            return str(result)

        except Exception as exc:
            message = str(exc)
            logger.error(
                f"[ChatService] Tool execution failed for '{tool_name}': {exc}",
                exc_info=True,
            )
            if self._is_auth_failure(message):
                await self._invalidate_github_access()
                return (
                    "Tool execution error: your GitHub access was rejected "
                    "(token expired or revoked). Please reconnect your GitHub account."
                )
            return f"Tool execution error: {exc}"

    async def _execute_tool_batch(self, tool_calls: list[dict]) -> List[ToolMessage]:
        """
        Execute multiple tool calls concurrently using asyncio.gather().
        Returns a list of ToolMessages in the same order as the input tool_calls.
        """
        async def _run_one(tc: dict) -> ToolMessage:
            tc_id = tc.get("id", "")
            tc_name = tc.get("name", "")
            tc_args = tc.get("args", {})

            if isinstance(tc_args, str):
                try:
                    tc_args = json.loads(tc_args)
                except json.JSONDecodeError:
                    tc_args = {}

            tool_output = await self._execute_tool(tc_name, tc_args)
            logger.info(
                f"[ChatService] Tool '{tc_name}' output (first 200 chars): {tool_output[:200]}"
            )
            return ToolMessage(content=tool_output, tool_call_id=tc_id)

        return await asyncio.gather(*[_run_one(tc) for tc in tool_calls])

    async def chat(self) -> AsyncIterator[str]:
        """
        Async generator — yields response text chunks token-by-token for
        StreamingResponse in the FastAPI route.
        """
        # ── Layer 1: Regex router — fast, zero-cost ───────────────────────────
        merged_registry = get_tool_registry()
        route_decision = route(
            self.user_prompt,
            merged_registry,
            conversation_history=self.conversation_history,
            router_llm=self.router_llm,
        )
        logger.info(f"[ChatService] {route_decision.description}")

        # Short-circuit: greeting or chitchat — no LLM call needed
        if route_decision.short_circuit and route_decision.greeting_response:
            yield route_decision.greeting_response
            return

        # ── Layer 2: LLM router for low-confidence decisions ──────────────────
        if route_decision.confidence == "low":
            # Skip LLM router for truly ambiguous queries (score=0, no regex match at all)
            # — the LLM router adds a full round-trip with minimal benefit.
            # Only invoke it for partial matches (score=1) where the router can disambiguate.
            is_partial_match = route_decision.description and "score=1" in route_decision.description
            if is_partial_match:
                logger.info("[ChatService] Low confidence (partial match) — invoking LLM router")
                route_decision = await route_with_llm(
                    decision=route_decision,
                    query=self.user_prompt,
                    tool_registry=merged_registry,
                    conversation_history=self.conversation_history,
                    router_llm=self.router_llm,
                )
                logger.info(f"[ChatService] LLM router decision: {route_decision.description}")
            else:
                logger.info("[ChatService] Low confidence (no match) — using default tools, skipping LLM router")

        selected_tools = route_decision.tools

        # GitHub intent: resolve THIS user's MCP connection (per-user OAuth token)
        github_client: MCPClient | None = None
        if route_decision.intent == Intent.GITHUB:
            if not self.user_id:
                logger.warning("[ChatService] GitHub intent but request has no user identity")
                yield "Please sign in first, then connect your GitHub account to use GitHub features."
                return

            github_client = await get_connected_client_for_user(self.user_id)
            if not github_client:
                logger.info(
                    f"[ChatService] GitHub intent but user {self.user_id} "
                    f"has no connected GitHub account"
                )
                yield (
                    "I can't access GitHub yet because no account is connected. "
                    "Please connect your GitHub account first (Settings → Connect GitHub), "
                    "then ask me again."
                )
                return

            selected_tools = list(github_client.mcp_tools)
            logger.info(
                f"[ChatService] GitHub intent — {len(selected_tools)} MCP tools bound "
                f"for user {self.user_id}"
            )

        # Per-request tool map: local tools + this user's MCP tools (if any)
        self.tool_map = dict(get_tool_registry())
        if github_client is not None:
            self.tool_map.update({t.name: t for t in github_client.mcp_tools})

        messages = [
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=self.user_prompt),
        ]

        parser = StrOutputParser()

        # Build tool-bound models — tool_choice="auto" makes the model decide
        # whether to call a tool or answer directly, using NATIVE function
        # calling (not text-based imitation).
        # If no tools selected, bind empty list so LLM answers directly.
        logger.info(f"[ChatService] Binding {len(selected_tools)} tools to LLM: {[t.name for t in selected_tools]}")
        primary_with_tools = self.primary_llm.bind_tools(
            selected_tools, tool_choice="auto"
        )
        fallback_with_tools = self.fallback_llm.bind_tools(
            selected_tools, tool_choice="auto"
        )

        # Create combined fallback chain for the first round (tool inspection)
        llm_with_tools_chain = primary_with_tools.with_fallbacks([fallback_with_tools])

        try:
            logger.info("[ChatService] Round 1: Invoking LLM with tool definitions.")

            # Multi-round tool-calling loop. Some workflows (e.g. GitHub:
            # get_me -> search_repositories) require MULTIPLE sequential tool
            # calls before the model can produce a final answer. A single
            # round of tool execution silently drops the follow-up calls and
            # yields an empty/generic response.
            MAX_TOOL_ROUNDS = 4
            tool_rounds_used = 0

            for _ in range(MAX_TOOL_ROUNDS):
                response: AIMessage = await llm_with_tools_chain.ainvoke(messages)

                tool_calls = getattr(response, "tool_calls", None)
                raw_content = (response.content or "").strip()

                logger.info(
                    f"[ChatService] Round {tool_rounds_used + 1} response — "
                    f"content ({len(raw_content)} chars): {raw_content[:300]!r}, "
                    f"tool_calls: {len(tool_calls) if tool_calls else 0}"
                )
                if tool_calls:
                    for tc in tool_calls:
                        logger.info(f"[ChatService]   -> tool_call: {tc.get('name')} args={tc.get('args')}")
                looks_like_fake_tool_call = raw_content.startswith('{"type": "function"') or (
                    raw_content.startswith("{")
                    and '"name"' in raw_content
                    and '"parameters"' in raw_content
                )

                if not tool_calls and looks_like_fake_tool_call:
                    logger.warning(
                        "[ChatService] Model emitted a fake text tool-call instead of "
                        "using native tool_calls. Retrying without tools on fallback LLM."
                    )
                    plain_chain = self.fallback_llm | parser
                    async for chunk in plain_chain.astream(messages):
                        if chunk:
                            yield chunk
                    return

                # ── Final text answer — stream it token-by-token ─────────────
                if not tool_calls:
                    logger.info("[ChatService] Final answer (no tool call)")
                    if tool_rounds_used == 0:
                        if raw_content:
                            yield raw_content
                        else:
                            stream_chain = (
                                self.primary_llm.with_fallbacks([self.fallback_llm]) | parser
                            )
                            async for chunk in stream_chain.astream(messages):
                                if chunk:
                                    yield chunk
                        return

                    # After tool use: re-stream with the bound chain so the
                    # final answer is delivered token-by-token.
                    stream_chain = llm_with_tools_chain | parser
                    chunk_count = 0
                    async for chunk in stream_chain.astream(messages):
                        if chunk:
                            yield chunk
                            chunk_count += 1
                    if chunk_count == 0:
                        logger.warning(
                            "[ChatService] Final streaming produced ZERO chunks — "
                            "falling back to direct content"
                        )
                        yield raw_content if raw_content else (
                            "I received the data but couldn't generate a response. "
                            "Please try rephrasing your question."
                        )
                    return

                # ── Tool call(s) detected — execute and continue the loop ────
                tool_rounds_used += 1
                tool_names = [tc.get("name", "") for tc in tool_calls]
                logger.info(f"[ChatService] Round {tool_rounds_used}: tool calls requested: {tool_names}")
                messages.append(response)

                tool_messages = await self._execute_tool_batch(tool_calls)
                messages.extend(tool_messages)

                for tm in tool_messages:
                    logger.info(f"[ChatService] Tool result ({len(tm.content)} chars): {tm.content[:300]!r}")

                # ── Skip an extra LLM round for short, self-contained tools ──
                # Tools like calculate_cgpa return complete answers — no need
                # for another LLM call to reformat them.
                _SKIP_ROUND2_TOOLS = {"calculate_cgpa"}
                all_skip = all(
                    tc.get("name", "") in _SKIP_ROUND2_TOOLS for tc in tool_calls
                )
                if all_skip and len(tool_messages) == 1:
                    result_text = tool_messages[0].content
                    if len(result_text) < 500:
                        logger.info("[ChatService] Skipping extra round — short self-contained tool result")
                        yield result_text
                        return

            # ── Max rounds reached — force a final answer from the model ─────
            logger.warning("[ChatService] Reached max tool rounds — forcing final answer")
            final_response: AIMessage = await llm_with_tools_chain.ainvoke(messages)
            final_content = (final_response.content or "").strip()
            if final_content:
                yield final_content
            else:
                yield "I couldn't complete that request. Please try rephrasing your question."

        except Exception as exc:
            logger.error(f"[ChatService] Chat failed entirely: {exc}", exc_info=True)
            yield "I'm currently experiencing technical difficulties. Please try again later."
