"""
mcp_client.py
-------------
Per-user lifecycle management of remote MCP server connections via
langchain-mcp-adapters.

Each user who connects their GitHub account gets an ISOLATED MCPClient —
their own OAuth access token in the Authorization header, so every tool
call executes as that user. Clients are cached by Clerk user id with
idle-TTL / capacity eviction.

Resilience (unchanged from the original design):
    - Startup connect: 3 attempts with exponential backoff (2s, 4s, 8s)
    - Lazy reconnect: throttled by a cooldown window, safe under concurrency
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient

logger = logging.getLogger("uvicorn")

# Startup retry policy — transient network/DNS blips shouldn't kill GitHub
# tools for the whole request.
_CONNECT_ATTEMPTS = 3
_CONNECT_BASE_DELAY = 2.0  # seconds — doubles each attempt (2s, 4s, 8s)

# Lazy reconnect throttle — at most one reconnect attempt per cooldown window,
# so a dead endpoint doesn't get hammered on every chat request.
_RECONNECT_COOLDOWN = 30.0  # seconds

# Client cache eviction
_MAX_CACHED_USERS = 50
_IDLE_TTL_SECONDS = 1800.0  # 30 minutes without use → evicted

_GITHUB_MCP_URL = "https://api.githubcopilot.com/mcp/"

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
    """Connection to the GitHub MCP server on behalf of ONE user."""

    def __init__(self, owner_id: str, access_token: str):
        self.owner_id = owner_id
        self.access_token = access_token
        self.client: MultiServerMCPClient | None = None
        self.mcp_tools: List[BaseTool] = []
        self._connected = False
        self._last_attempt = 0.0  # monotonic timestamp of last connect attempt
        self.last_used = time.monotonic()
        self._connect_lock = asyncio.Lock()  # serialize concurrent reconnects

    @property
    def is_connected(self) -> bool:
        return self._connected and len(self.mcp_tools) > 0

    def mark_used(self) -> None:
        self.last_used = time.monotonic()

    async def connect(self):
        """Connect to MCP servers with retries + exponential backoff."""
        for attempt in range(1, _CONNECT_ATTEMPTS + 1):
            try:
                await self._connect_once()
                return
            except Exception as exc:
                logger.error(
                    f"[MCP] Connection attempt {attempt}/{_CONNECT_ATTEMPTS} "
                    f"failed for user {self.owner_id}: {exc}"
                )
                if attempt < _CONNECT_ATTEMPTS:
                    delay = _CONNECT_BASE_DELAY * (2 ** (attempt - 1))
                    logger.warning(f"[MCP] Retrying in {delay:.0f}s...")
                    await asyncio.sleep(delay)

        logger.error(
            "[MCP] All connection attempts failed — GitHub tools UNAVAILABLE for "
            f"user {self.owner_id}. Lazy reconnect will be attempted on the next "
            "GitHub query. Check that the user's OAuth token is valid and "
            f"{_GITHUB_MCP_URL} is reachable."
        )

    async def ensure_connected(self) -> bool:
        """
        Lazy reconnect for runtime recovery. Returns True if tools are (now)
        available. Throttled by _RECONNECT_COOLDOWN and safe under concurrency.
        """
        if self.is_connected:
            self.mark_used()
            return True

        async with self._connect_lock:
            # Re-check after acquiring the lock — another task may have reconnected
            if self.is_connected:
                self.mark_used()
                return True

            now = time.monotonic()
            if now - self._last_attempt < _RECONNECT_COOLDOWN:
                logger.info("[MCP] Lazy reconnect skipped — cooldown window active")
                return False

            logger.info("[MCP] No MCP tools loaded — attempting lazy reconnect...")
            await self.connect()
            if self.is_connected:
                logger.info("[MCP] Lazy reconnect succeeded")
            self.mark_used()
            return self.is_connected

    async def _connect_once(self):
        """Single connection attempt — raises on failure."""
        config = {
            "github": {
                "transport": "streamable_http",
                "url": _GITHUB_MCP_URL,
                "headers": {
                    "Authorization": f"Bearer {self.access_token}",
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
                f"[MCP] Connected to GitHub MCP for user {self.owner_id} — "
                f"{len(self.mcp_tools)} essential tools loaded ({dropped} filtered out)"
            )
        except Exception as exc:
            self.client = None
            self.mcp_tools = []
            self._connected = False
            # Failure starts the cooldown window for lazy reconnects
            self._last_attempt = time.monotonic()
            raise

    async def disconnect(self):
        self.client = None
        self.mcp_tools = []
        self._connected = False


class MCPClientManager:
    """
    Cache of per-user MCP clients: clerk_user_id → MCPClient.

    All public methods are safe under concurrency. Eviction happens on new
    connections: idle entries past TTL are dropped first, then the
    least-recently-used when over capacity.
    """

    def __init__(self):
        self._clients: Dict[str, MCPClient] = {}
        self._lock = asyncio.Lock()

    async def get_or_connect(self, user_id: str, access_token: str) -> MCPClient:
        """
        Return a connected client for the user — reusing the cached one when
        possible, creating a fresh connection otherwise (e.g. first use after
        OAuth connect, or after eviction).
        """
        async with self._lock:
            existing = self._clients.get(user_id)
            if existing is not None and existing.access_token == access_token:
                self._evict_locked()
                connected = await existing.ensure_connected()
                return existing if connected else None
            # New user or re-auth with a different token — replace the entry
            self._clients[user_id] = MCPClient(user_id, access_token)
            self._evict_locked()

        client = self._clients[user_id]
        await client.connect()
        return client if client.is_connected else None

    async def get_cached(self, user_id: str) -> Optional[MCPClient]:
        """Return the cached client if present and connected; never connects."""
        async with self._lock:
            client = self._clients.get(user_id)
            if client is None:
                return None
            self._evict_locked()
        if await client.ensure_connected():
            return client
        return None

    def disconnect_user(self, user_id: str) -> None:
        """Drop the user's cached client (e.g. after disconnect/revocation)."""
        client = self._clients.pop(user_id, None)
        if client:
            logger.info(f"[MCP] Disconnected MCP client for user {user_id}")

    def stats(self) -> dict:
        connected = sum(1 for c in self._clients.values() if c.is_connected)
        return {"cached_users": len(self._clients), "connected_users": connected}

    def _evict_locked(self) -> None:
        """Must be called while holding self._lock."""
        now = time.monotonic()
        expired = [
            uid
            for uid, c in self._clients.items()
            if now - c.last_used > _IDLE_TTL_SECONDS
        ]
        for uid in expired:
            del self._clients[uid]
            logger.info(f"[MCP] Evicted idle MCP client for user {uid}")

        while len(self._clients) >= _MAX_CACHED_USERS:
            oldest = min(self._clients, key=lambda u: self._clients[u].last_used)
            del self._clients[oldest]
            logger.info(f"[MCP] Evicted LRU MCP client for user {oldest}")


async def get_connected_client_for_user(user_id: str) -> Optional[MCPClient]:
    """
    Resolve the user's stored OAuth token and return a connected MCP client,
    or None if the user hasn't connected GitHub (or the connection fails).
    Convenience bridge used by ChatService.
    """
    if not user_id:
        return None
    from app.db.token_store import get_decrypted_token

    token = await get_decrypted_token(user_id)
    if not token:
        return None
    return await mcp_manager.get_or_connect(user_id, token)


mcp_manager = MCPClientManager()
