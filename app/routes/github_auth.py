"""
routes/github_auth.py
---------------------
Endpoints for per-user GitHub account connection (OAuth App flow).

    GET    /auth/github/login      → {authorize_url} for the frontend redirect
    POST   /auth/github/callback   → exchange code, store encrypted token
    GET    /auth/github/status     → {connected, github_login}
    DELETE /auth/github/disconnect → revoke grant on GitHub, remove stored
                                    token, evict MCP client

All endpoints require a valid Clerk session (get_current_user).
"""

import logging

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from app.auth.dependencies import get_current_user
from app.auth import github_oauth
from app.db import token_store
from app.services.mcp_client import mcp_manager

logger = logging.getLogger("uvicorn")

router = APIRouter(prefix="/auth/github", tags=["github-auth"])


class GitHubCallbackRequest(BaseModel):
    code: str
    state: str


@router.get("/login")
async def github_login(current_user: dict = Depends(get_current_user)):
    """Return the GitHub consent-screen URL for the logged-in user."""
    authorize_url = github_oauth.build_authorize_url(current_user["sub"])
    return {"authorize_url": authorize_url}


@router.post("/callback")
async def github_callback(
    body: GitHubCallbackRequest,
    current_user: dict = Depends(get_current_user),
):
    """
    Complete the OAuth round-trip: validate the signed state (CSRF guard),
    exchange the code for an access token, store it encrypted, and warm up
    the user's MCP connection.
    """
    clerk_user_id = current_user["sub"]

    # The state must have been issued for THIS Clerk user
    state_user = github_oauth.verify_state(body.state)
    if state_user != clerk_user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="GitHub sign-in state does not match the logged-in user.",
        )

    access_token, scopes = github_oauth.exchange_code_for_token(body.code)
    github_login = github_oauth.fetch_github_login(access_token)

    await token_store.save_token(clerk_user_id, access_token, github_login, scopes)

    # Warm the MCP connection so the first GitHub query is fast.
    # Failure here is non-fatal — lazy reconnect will retry on first use.
    try:
        await mcp_manager.get_or_connect(clerk_user_id, access_token)
    except Exception as exc:
        logger.warning(f"[GitHubAuth] MCP warm-up failed for {clerk_user_id}: {exc}")

    return {"connected": True, "github_login": github_login}


@router.get("/status")
async def github_status(current_user: dict = Depends(get_current_user)):
    """Whether the logged-in user has connected their GitHub account."""
    github_login = await token_store.get_github_login(current_user["sub"])
    return {"connected": github_login is not None, "github_login": github_login}


@router.delete("/disconnect")
async def github_disconnect(current_user: dict = Depends(get_current_user)):
    """
    Full disconnect for the logged-in user only:
      1. Revoke the OAuth grant/token on GitHub's side (so reconnecting
         starts a fresh consent flow and can't silently reuse the old
         account).
      2. Delete the encrypted token + GitHub info from the database.
      3. Drop the user's cached MCP client.

    GitHub-side revocation is best-effort: if GitHub is unreachable the
    local connection is still removed, and the stale remote token dies on
    its own expiry / can be revoked from github.com settings.
    """
    clerk_user_id = current_user["sub"]

    revoked = False
    access_token = await token_store.get_decrypted_token(clerk_user_id)
    if access_token:
        revoked = github_oauth.revoke_user_authorization(access_token)

    deleted = await token_store.delete_token(clerk_user_id)
    mcp_manager.disconnect_user(clerk_user_id)
    return {"disconnected": deleted, "revoked": revoked}
