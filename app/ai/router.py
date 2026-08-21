"""
router.py
---------
Hybrid intent classifier and tool-subset selector.

Layer 1 — Regex-based intent classification (zero-cost, fast).
  - High-confidence match (score >= 2): routes directly.
  - Low-confidence / ambiguous: falls through to Layer 2.

Layer 2 — LLM-based intent classification (Gemini Flash, fast).
  - Used only when Layer 1 is ambiguous or low-confidence.
  - Receives conversation history for context-aware routing.
  - Classifies into a structured JSON response with intent + tool list.

The LLM still makes the final tool-calling decision, but from a smaller,
more relevant set of tools — reducing token usage, improving accuracy,
and enabling short-circuits for simple queries.
"""

import re
import json
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional

from langchain_core.tools import BaseTool

logger = logging.getLogger("uvicorn")


# ──────────────────────────────────────────────────────────────────────────────
# Intent categories
# ──────────────────────────────────────────────────────────────────────────────
class Intent:
    GREETING = "greeting"
    CGPA = "cgpa"
    SYLLABUS = "syllabus"
    WEB_SEARCH = "web_search"
    GITHUB = "github"
    AMBIGUOUS = "ambiguous"


# Confidence thresholds
HIGH_CONFIDENCE = 2
LOW_CONFIDENCE = 1


# ──────────────────────────────────────────────────────────────────────────────
# Regex patterns — order matters for priority (first match wins at same score)
# ──────────────────────────────────────────────────────────────────────────────
_PATTERNS: Dict[str, re.Pattern] = {
    Intent.GREETING: re.compile(
        r"^\s*(hi|hello|hey|howdy|sup|yo|thanks?|thank you|bye|goodbye|"
        r"good\s*(morning|afternoon|evening)|what'?s\s+up|how\s+are\s+you"
        r"|help|who\s+are\s+you|what\s+can\s+you\s+do)\s*[!?.]*\s*$",
        re.IGNORECASE,
    ),
    Intent.CGPA: re.compile(
        r"\b(calculate|cgpa|gpa|sgpa|grade|cumulative|semester\s*gpas?|"
        r"my\s+gpas?|sem\s*gpas?|average\s*grade)\b",
        re.IGNORECASE,
    ),
    Intent.SYLLABUS: re.compile(
        r"\b(syllabus|subject|course|unit|module|credits?|semester|sem\s|"
        r"objectives?|outcomes?|topics?|taught|offered|curriculum|lab|practical|"
        r"theory|department|regulation|cbc|22l[a-z]{2}\d{3})\b",
        re.IGNORECASE,
    ),
    Intent.WEB_SEARCH: re.compile(
        r"\b(latest|current|trend|placement|news|2024|2025|2026|"
        r"gate|exam\s+pattern|industry|interview|career|job|"
        r"recruitment|package|salary|company)\b",
        re.IGNORECASE,
    ),
    Intent.GITHUB: re.compile(
        r"\b(github|repos?(?:itories)?|repository|repositories|issue|pull\s*request|\bpr\b|commit|"
        r"branch|merge|fork|clone|push|pull|code|readme|license|gists?|milestones?|starred)\b|"
        r"(?=(?:search|list|show|find|get)\s+(?:for\s+)?(?:my\s+)?repos?(?:itories)?\b)",
        re.IGNORECASE,
    ),
}


# ──────────────────────────────────────────────────────────────────────────────
# Route decision dataclass
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class RouteDecision:
    """Result of the routing step — passed to ChatService."""
    intent: str
    tools: List[BaseTool]
    short_circuit: bool = False
    greeting_response: Optional[str] = None
    description: str = ""
    confidence: str = "high"  # "high" | "low" | "llm"


# ──────────────────────────────────────────────────────────────────────────────
# Layer 1: Regex-based intent classifier
# ──────────────────────────────────────────────────────────────────────────────
def classify_intent(query: str) -> tuple[str, int]:
    """
    Classify the user query into an intent using regex pattern matching.

    Returns (intent, score) where score indicates match strength.
    """
    scores: Dict[str, int] = {k: 0 for k in _PATTERNS}

    for intent, pattern in _PATTERNS.items():
        matches = pattern.findall(query)
        if matches:
            scores[intent] += len(matches)

    # Intent priority: higher value wins ties. GitHub gets priority because
    # GitHub queries often mention other keywords (e.g. "repo", "code") that
    # can trigger syllabus/web_search intents as side effects.
    _INTENT_PRIORITY = {
        Intent.GITHUB: 10,
        Intent.CGPA: 5,
        Intent.SYLLABUS: 4,
        Intent.WEB_SEARCH: 3,
        Intent.GREETING: 2,
        Intent.AMBIGUOUS: 0,
    }

    best_intent = max(
        scores,
        key=lambda k: (scores[k], _INTENT_PRIORITY.get(k, 0)),
    )
    best_score = scores[best_intent]

    if best_score == 0:
        return Intent.AMBIGUOUS, 0

    logger.info(f"[Router-L1] Intent: {best_intent} (score: {best_score}, scores: {scores})")
    return best_intent, best_score


# ──────────────────────────────────────────────────────────────────────────────
# Tool subset selector (rule-based)
# ──────────────────────────────────────────────────────────────────────────────
_INTENT_TOOL_MAP = {
    Intent.CGPA: ["calculate_cgpa"],
    Intent.SYLLABUS: ["syllabus_rag_search"],
    Intent.WEB_SEARCH: ["web_search_tool"],
    Intent.GITHUB: [],  # MCP tool names are dynamic — LLM layer selects them
    Intent.AMBIGUOUS: ["syllabus_rag_search", "web_search_tool"],
}

_GREETING_RESPONSES = [
    "Hello! I'm your ECE department assistant. "
    "I can help you with syllabus information, CGPA calculations, "
    "current ECE-related news, and GitHub operations. What would you like to know?",
]


def _select_tools_by_intent(intent: str, tool_registry: Dict[str, BaseTool]) -> List[BaseTool]:
    """Map an intent string to a list of tool instances from the registry."""
    tool_names = _INTENT_TOOL_MAP.get(intent, _INTENT_TOOL_MAP[Intent.AMBIGUOUS])

    # GitHub intent: include ALL non-local tools from the registry (MCP tools are dynamic)
    _LOCAL_TOOL_NAMES = {"calculate_cgpa", "syllabus_rag_search", "web_search_tool"}
    if intent == Intent.GITHUB:
        github_tools = [t for t in tool_registry.values() if t.name not in _LOCAL_TOOL_NAMES]
        if github_tools:
            logger.info(f"[Router] GitHub intent — {len(github_tools)} MCP tools from registry")
            return github_tools
        # No MCP tools loaded — will be caught by chat_service guard
        logger.warning("[Router] GitHub intent but no MCP tools in registry")
        return []

    selected = [tool_registry[name] for name in tool_names if name in tool_registry]
    if not selected:
        selected = list(tool_registry.values())
        logger.warning(f"[Router] No tools matched intent '{intent}', falling back to all tools")
    return selected


# ──────────────────────────────────────────────────────────────────────────────
# Layer 2: LLM-based intent classifier
# ──────────────────────────────────────────────────────────────────────────────
_ROUTER_LLM_PROMPT = """You are an intent classifier for a college ECE department chatbot.

Classify the user's query into one of these categories and select the best tool(s).

## Available Tools
- calculate_cgpa: CGPA/GPA calculations, grade computations
- syllabus_rag_search: Course content, subjects, units, credits, objectives, syllabus
- web_search_tool: Current news, placements, exam patterns, career trends, industry info
- GitHub tools (MCP): search_repositories, create_repository, get_file_contents, list_issues, issue_read, issue_write, list_pull_requests, create_pull_request, list_commits, list_branches, search_code, get_me, and more. Use intent "github" for any GitHub-related operation.

## Conversation History
{history_context}

## Rules
1. If the query is a greeting or chitchat, respond with intent "greeting".
2. If the query clearly matches ONE tool, use that tool.
3. If the query involves GitHub (repos, issues, PRs, commits, branches), use intent "github" and select the appropriate GitHub tool(s) by name.
4. If the query could involve multiple tools (e.g., comparing syllabus with current trends), list multiple tools in priority order.
5. Consider conversation history for ambiguous follow-ups (e.g., "what about semester 4?" after a CGPA question).
6. Prefer syllabus_rag_search for academic/college-specific questions.
7. ANY query mentioning "github", "repo", "repositor*", issues, PRs, commits, branches, README, license, or repo files MUST use intent "github" — even if it also contains "search", "list", "show", or "find". Do NOT classify such queries as web_search or syllabus.

Respond in EXACTLY this JSON format — no other text:
{{"intent": "<intent_name>", "tools": ["<tool_name>", ...], "reasoning": "<brief reason>"}}

User query: {query}"""


async def _llm_classify(
    query: str,
    history_context: str,
    router_llm,
) -> Optional[dict]:
    """
    Use an LLM to classify the query when regex is ambiguous.
    Returns {"intent": str, "tools": [str, ...]} or None on failure.
    """
    prompt = _ROUTER_LLM_PROMPT.format(
        query=query,
        history_context=history_context or "No prior conversation.",
    )

    try:
        from langchain_core.messages import HumanMessage
        response = await router_llm.ainvoke([HumanMessage(content=prompt)])
        content = response.content
        # Gemini returns content as a list of parts (dicts or objects) — normalize to text.
        if isinstance(content, list):
            texts = []
            for part in content:
                if isinstance(part, dict):
                    text = part.get("text") or part.get("content") or ""
                else:
                    text = getattr(part, "text", None) or ""
                texts.append(str(text))
            raw = "".join(texts)
        else:
            raw = str(content)
        raw = raw.strip()

        # Extract JSON from possible markdown fences
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            # Fallback: grab the first {...} block if the model wrapped the JSON
            # with extra text.
            start = raw.find("{")
            end = raw.rfind("}")
            if start != -1 and end > start:
                parsed = json.loads(raw[start : end + 1])
            else:
                raise
        logger.info(f"[Router-L2] LLM classified: {parsed}")
        return parsed

    except Exception as exc:
        logger.warning(f"[Router-L2] LLM classification failed: {exc}")
        return None


# ──────────────────────────────────────────────────────────────────────────────
# Main routing function
# ──────────────────────────────────────────────────────────────────────────────
def route(
    query: str,
    tool_registry: Dict[str, BaseTool],
    conversation_history: Optional[List[dict]] = None,
    router_llm=None,
) -> RouteDecision:
    """
    Classify the user query and select the appropriate tool subset.

    Parameters
    ----------
    query               : The user's input text.
    tool_registry       : Dict mapping tool name -> BaseTool instance.
    conversation_history: Optional list of prior messages for context-aware routing.
    router_llm          : Optional LLM instance for Layer 2 fallback.

    Returns
    -------
    RouteDecision with intent, tool subset, confidence, and whether to short-circuit.
    """
    # ── Layer 1: Regex classification ─────────────────────────────────────
    intent, score = classify_intent(query)

    # Short-circuit for greetings — no LLM call needed
    if intent == Intent.GREETING:
        greeting = _GREETING_RESPONSES[0]
        logger.info(f"[Router] Short-circuiting for greeting: '{query[:50]}'")
        return RouteDecision(
            intent=intent,
            tools=[],
            short_circuit=True,
            greeting_response=greeting,
            description="Greeting detected — direct response, no LLM needed",
            confidence="high",
        )

    # High-confidence regex match — route directly
    if score >= HIGH_CONFIDENCE:
        selected_tools = _select_tools_by_intent(intent, tool_registry)
        logger.info(
            f"[Router] High confidence '{intent}' (score={score}) -> "
            f"{[t.name for t in selected_tools]}"
        )
        return RouteDecision(
            intent=intent,
            tools=selected_tools,
            description=f"Regex high-confidence '{intent}' (score={score})",
            confidence="high",
        )

    # ── Layer 1 low-confidence or ambiguous → try Layer 2 ─────────────────
    if score == LOW_CONFIDENCE or intent == Intent.AMBIGUOUS:
        if router_llm is not None:
            # Build conversation history context
            history_context = _build_history_context(conversation_history)

            # Import inside function to avoid circular imports
            llm_result = None
            # llm_result is awaited via a helper; but route() is sync,
            # so we store the coroutine and let the caller handle it.
            # Actually, to keep route() sync-compatible, we use a different
            # pattern: the caller passes router_llm and we do the async call
            # only if the caller is async. We handle this by returning a
            # special marker and letting ChatService do the async LLM call.
            #
            # For simplicity, we use a synchronous approach: if router_llm is
            # provided and we're in an async context, we'll be called from
            # an async method. We store the intent/tools for the caller.
            logger.info(
                f"[Router] Low confidence '{intent}' (score={score}), "
                f"deferring to LLM router"
            )
            return RouteDecision(
                intent=intent,
                tools=_select_tools_by_intent(intent, tool_registry),
                description=(
                    f"Regex low-confidence '{intent}' (score={score}) — "
                    f"LLM router available, caller should invoke it"
                ),
                confidence="low",
            )

        # No LLM router available — fall back to rule-based
        selected_tools = _select_tools_by_intent(intent, tool_registry)
        logger.info(
            f"[Router] Low confidence '{intent}' (score={score}), "
            f"no LLM router -> rule-based fallback"
        )
        return RouteDecision(
            intent=intent,
            tools=selected_tools,
            description=f"Regex low-confidence '{intent}' (score={score}), rule-based fallback",
            confidence="low",
        )

    # Fallback: score=0 and not a greeting (shouldn't happen, but safety net)
    selected_tools = _select_tools_by_intent(Intent.AMBIGUOUS, tool_registry)
    return RouteDecision(
        intent=Intent.AMBIGUOUS,
        tools=selected_tools,
        description="No regex match — using all ambiguous tools",
        confidence="low",
    )


def _build_history_context(conversation_history: Optional[List[dict]]) -> str:
    """Format conversation history into a readable string for the LLM router."""
    if not conversation_history:
        return "No prior conversation."

    lines = []
    for msg in conversation_history[-6:]:  # Last 3 exchanges max
        role = msg.get("role", "unknown")
        content = msg.get("content", "")[:150]  # Truncate long messages
        lines.append(f"{role.capitalize()}: {content}")
    return "\n".join(lines)


async def route_with_llm(
    decision: RouteDecision,
    query: str,
    tool_registry: Dict[str, BaseTool],
    conversation_history: Optional[List[dict]],
    router_llm,
) -> RouteDecision:
    """
    Async helper: invoke the LLM router for low-confidence decisions.
    Called from ChatService when confidence == "low" and router_llm is available.

    Falls back to the original regex-based decision if the LLM fails.
    """
    history_context = _build_history_context(conversation_history)
    parsed = await _llm_classify(query, history_context, router_llm)

    if parsed is None:
        # LLM failed — return original decision
        logger.info("[Router] LLM classification failed, using original regex decision")
        return decision

    # Map LLM tool names to actual tool instances
    llm_tool_names = parsed.get("tools", [])
    selected_tools = []
    for name in llm_tool_names:
        if name in tool_registry:
            selected_tools.append(tool_registry[name])

    if not selected_tools:
        logger.warning("[Router] LLM returned no valid tools, using original decision")
        return decision

    llm_intent = parsed.get("intent", decision.intent)
    reasoning = parsed.get("reasoning", "")

    logger.info(
        f"[Router-L2] Final decision: intent='{llm_intent}', "
        f"tools={[t.name for t in selected_tools]}, reason='{reasoning}'"
    )

    return RouteDecision(
        intent=llm_intent,
        tools=selected_tools,
        description=f"LLM router: '{llm_intent}' — {reasoning}",
        confidence="llm",
    )
