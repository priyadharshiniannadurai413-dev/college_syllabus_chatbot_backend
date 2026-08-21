"""
mcp_client.py
-------------
Manages the lifecycle of remote MCP servers via langchain-mcp-adapters.

On startup, connects to configured MCP servers (GitHub) and fetches
their tool schemas. Filters to essential toolsets to reduce token usage.
"""

import asyncio
import logging
import time
from typing import List

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient

from app.core.config import settings

logger = logging.getLogger("uvicorn")

# Startup retry policy — transient network/DNS blips shouldn't kill GitHub
# tools for the whole process lifetime.
_CONNECT_ATTEMPTS = 3
_CONNECT_BASE_DELAY = 2.0  # seconds — doubles each attempt (2s, 4s, 8s)

# Lazy reconnect throttle — at most one reconnect attempt per cooldown window,
# so a dead endpoint doesn't get hammered on every chat request.
_RECONNECT_COOLDOWN = 30.0  # seconds

# Essential GitHub tool names — keeps token cost ~12k instead of ~28k.
# Covers repos, issues, PRs, code search, branches, commits, and user info.
_ESSENTIAL_TOOLS = {
    # Repos
    "search_repositories", "create_repository", "fork_repository",
    "get_file_contents", "create_or_update_file", "delete_file", "push_files",
    # Branches
    "list_branches", "create_branch",
    # Commits
    "list_commits", "get_commit", "search_commits",
    # Issues
    "list_issues", "issue_read", "issue_write", "add_issue_comment",
    "search_issues", "list_issue_fields", "list_issue_types", "sub_issue_write",
    # Pull Requests
    "list_pull_requests", "pull_request_read", "create_pull_request",
    "update_pull_request", "update_pull_request_branch", "merge_pull_request",
    "pull_request_review_write", "add_reply_to_pull_request_comment",
    "request_copilot_review", "search_pull_requests",
    # Code search
    "search_code",
    # User
    "get_me",
}


class MCPClient:
    def __init__(self):
        self.client: MultiServerMCPClient | None = None
        self.mcp_tools: List[BaseTool] = []
        self._connected = False
        self._last_attempt = 0.0  # monotonic timestamp of last connect attempt
        self._connect_lock = asyncio.Lock()  # serialize concurrent reconnects

    @property
    def is_connected(self) -> bool:
        return self._connected and len(self.mcp_tools) > 0

    async def connect(self):
        """Connect to MCP servers with retries + exponential backoff."""
        for attempt in range(1, _CONNECT_ATTEMPTS + 1):
            try:
                await self._connect_once()
                return
            except Exception as exc:
                logger.error(
                    f"[MCP] Connection attempt {attempt}/{_CONNECT_ATTEMPTS} failed: {exc}"
                )
                if attempt < _CONNECT_ATTEMPTS:
                    delay = _CONNECT_BASE_DELAY * (2 ** (attempt - 1))
                    logger.warning(f"[MCP] Retrying in {delay:.0f}s...")
                    await asyncio.sleep(delay)

        logger.error(
            "[MCP] All connection attempts failed — GitHub tools UNAVAILABLE. "
            "Lazy reconnect will be attempted on the next GitHub query. "
            "Check: 1) GITHUB_API_KEY is a valid PAT, "
            "2) api.githubcopilot.com is reachable, "
            "3) langchain-mcp-adapters is installed correctly."
        )

    async def ensure_connected(self) -> bool:
        """
        Lazy reconnect for runtime recovery. Returns True if tools are (now)
        available. Throttled by _RECONNECT_COOLDOWN and safe under concurrency.
        """
        if self.is_connected:
            return True

        async with self._connect_lock:
            # Re-check after acquiring the lock — another task may have reconnected
            if self.is_connected:
                return True

            now = time.monotonic()
            if now - self._last_attempt < _RECONNECT_COOLDOWN:
                logger.info("[MCP] Lazy reconnect skipped — cooldown window active")
                return False

            logger.info("[MCP] No MCP tools loaded — attempting lazy reconnect...")
            await self.connect()
            if self.is_connected:
                logger.info("[MCP] Lazy reconnect succeeded")
            return self.is_connected

    async def _connect_once(self):
        """Single connection attempt — raises on failure."""
        if not settings.GITHUB_PAT:
            logger.warning("[MCP] GITHUB_PAT not set — skipping MCP server connection. GitHub tools will be unavailable.")
            return

        # Validate token format (should start with ghp_ or github_pat_)
        if not (settings.GITHUB_PAT.startswith("ghp_") or settings.GITHUB_PAT.startswith("github_pat_")):
            logger.warning(
                f"[MCP] GITHUB_PAT has unexpected format: '{settings.GITHUB_PAT[:10]}...' "
                f"(expected 'ghp_' or 'github_pat_' prefix). Connection may fail."
            )

        config = {
            "github": {
                "transport": "streamable_http",
                "url": "https://api.githubcopilot.com/mcp/",
                "headers": {
                    "Authorization": f"Bearer {settings.GITHUB_PAT}",
                },
            },
        }

        try:
            self.client = MultiServerMCPClient(config)
            all_tools = await self.client.get_tools()

            # Filter to essential toolsets only — reduces ~28k tokens to ~12k
            self.mcp_tools = [t for t in all_tools if t.name in _ESSENTIAL_TOOLS]
            dropped = len(all_tools) - len(self.mcp_tools)
            self._connected = True
            # Success resets the throttle clock — a later mid-session drop
            # should be recoverable immediately, not blocked by cooldown.
            self._last_attempt = 0.0
            logger.info(
                f"[MCP] Connected to GitHub MCP — {len(self.mcp_tools)} essential tools loaded "
                f"({dropped} filtered out)"
            )
            logger.info(f"[MCP] Tools: {[t.name for t in self.mcp_tools]}")
            # Invalidate cached tool registries so they rebuild with MCP tools
            from app.tools import invalidate_tool_cache
            invalidate_tool_cache()
        except Exception as exc:
            self.client = None
            self.mcp_tools = []
            self._connected = False
            # Failure starts the cooldown window for lazy reconnects
            self._last_attempt = time.monotonic()
            from app.tools import invalidate_tool_cache
            invalidate_tool_cache()
            raise

    async def disconnect(self):
        if self.client:
            logger.info("[MCP] Disconnected from MCP servers")
            self.client = None
            self.mcp_tools = []
            self._connected = False

    def get_tools(self) -> List[BaseTool]:
        return self.mcp_tools


mcp_client = MCPClient()
