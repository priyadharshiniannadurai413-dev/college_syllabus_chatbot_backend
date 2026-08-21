"""
github_oauth.py
---------------
GitHub OAuth App — web application flow helpers.

Flow:
    1. build_authorize_url(clerk_user_id)
       → frontend redirects the user's browser to GitHub's consent screen.
    2. User authorizes → GitHub redirects to GITHUB_OAUTH_REDIRECT_URI
       (frontend page) with ?code=...&state=...
    3. Frontend POSTs {code, state} to /auth/github/callback.
    4. verify_state(state) binds the round-trip to the Clerk user (CSRF guard),
       then exchange_code_for_token(code) + fetch_github_login(token).

The `state` parameter is a short-lived HS256 JWT signed with
TOKEN_ENCRYPTION_KEY and embedding the Clerk user id — this proves the
OAuth round-trip was initiated by the same logged-in user that finishes it.
"""

import logging
from datetime import datetime, timedelta, timezone
from urllib.parse import urlencode

import requests
from fastapi import HTTPException, status
from jose import ExpiredSignatureError, JWTError, jwt

from app.core.config import settings

logger = logging.getLogger("uvicorn")

_GITHUB_AUTHORIZE_URL = "https://github.com/login/oauth/authorize"
_GITHUB_TOKEN_URL = "https://github.com/login/oauth/access_token"
_GITHUB_USER_API = "https://api.github.com/user"

# Server-side revocation endpoints (Applications API). Authenticated with the
# app's own client credentials via HTTP Basic — the user's token is sent only
# in the JSON body over HTTPS and is never logged.
_GITHUB_GRANT_DELETE_URL = "https://api.github.com/applications/{client_id}/grant"
_GITHUB_TOKEN_REVOKE_URL = "https://api.github.com/applications/{client_id}/token"

# Requested scopes: full repo access (issues, PRs, commits, file read/write)
# plus basic profile read for display purposes.
_GITHUB_SCOPES = "repo read:user"

_STATE_TTL_MINUTES = 10


def _signing_key() -> bytes:
    if not settings.TOKEN_ENCRYPTION_KEY:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="TOKEN_ENCRYPTION_KEY is not configured.",
        )
    return settings.TOKEN_ENCRYPTION_KEY.encode()


def _require_oauth_config() -> None:
    missing = [
        name
        for name in (
            "GITHUB_OAUTH_CLIENT_ID",
            "GITHUB_OAUTH_CLIENT_SECRET",
            "GITHUB_OAUTH_REDIRECT_URI",
        )
        if not getattr(settings, name)
    ]
    if missing:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"GitHub OAuth is not configured. Missing: {', '.join(missing)}",
        )


def build_authorize_url(clerk_user_id: str) -> str:
    """Build the GitHub consent-screen URL with a signed, user-bound state."""
    _require_oauth_config()
    state = jwt.encode(
        {
            "sub": clerk_user_id,
            "exp": datetime.now(timezone.utc) + timedelta(minutes=_STATE_TTL_MINUTES),
        },
        _signing_key(),
        algorithm="HS256",
    )
    params = urlencode(
        {
            "client_id": settings.GITHUB_OAUTH_CLIENT_ID,
            "redirect_uri": settings.GITHUB_OAUTH_REDIRECT_URI,
            "scope": _GITHUB_SCOPES,
            "state": state,
        }
    )
    return f"{_GITHUB_AUTHORIZE_URL}?{params}"


def verify_state(state: str) -> str:
    """
    Validate the signed state and return the Clerk user id it was issued for.
    Raises 400 on tampered/expired states.
    """
    try:
        payload = jwt.decode(state, _signing_key(), algorithms=["HS256"])
        clerk_user_id = payload.get("sub")
        if not clerk_user_id:
            raise JWTError("state payload missing sub")
        return clerk_user_id
    except ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="GitHub sign-in link expired. Please try connecting again.",
        )
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid GitHub sign-in state.",
        )


def exchange_code_for_token(code: str) -> tuple[str, str]:
    """
    Exchange the OAuth code for an access token.
    Returns (access_token, scopes).
    """
    _require_oauth_config()
    try:
        resp = requests.post(
            _GITHUB_TOKEN_URL,
            json={
                "client_id": settings.GITHUB_OAUTH_CLIENT_ID,
                "client_secret": settings.GITHUB_OAUTH_CLIENT_SECRET,
                "code": code,
                "redirect_uri": settings.GITHUB_OAUTH_REDIRECT_URI,
            },
            headers={"Accept": "application/json"},
            timeout=15,
        )
        resp.raise_for_status()
    except requests.RequestException as exc:
        logger.error(f"[GitHubOAuth] Token exchange request failed: {exc}")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Could not reach GitHub to complete sign-in. Please retry.",
        )

    data = resp.json()
    access_token = data.get("access_token")
    if not access_token:
        error = data.get("error_description") or data.get("error") or "unknown error"
        logger.warning(f"[GitHubOAuth] Token exchange rejected: {error}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"GitHub rejected the sign-in: {error}",
        )

    # GitHub returns scopes as a comma-separated string (e.g. "repo,read:user")
    scopes = data.get("scope") or ""
    if isinstance(scopes, list):
        scopes = ",".join(scopes)
    return access_token, scopes


def fetch_github_login(access_token: str) -> str:
    """Fetch the authenticated user's GitHub username for display."""
    try:
        resp = requests.get(
            _GITHUB_USER_API,
            headers={
                "Authorization": f"Bearer {access_token}",
                "Accept": "application/vnd.github+json",
            },
            timeout=15,
        )
        resp.raise_for_status()
        return resp.json()["login"]
    except (requests.RequestException, KeyError) as exc:
        logger.error(f"[GitHubOAuth] Failed to fetch GitHub user: {exc}")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Signed in with GitHub but could not fetch your username.",
        )


def revoke_user_authorization(access_token: str) -> bool:
    """
    Revoke the user's OAuth grant on GitHub's side so reconnecting always
    starts a fresh consent flow instead of silently reusing the previous
    account.

    Tries deleting the authorization grant first (forces GitHub to re-consent
    on reconnect); falls back to plain token revocation if grant deletion is
    not supported for this app. Returns True when either succeeded.

    Note: this cannot sign the user out of their github.com browser session —
    that session belongs to GitHub and no API exists to end it.
    """
    _require_oauth_config()
    basic = (settings.GITHUB_OAUTH_CLIENT_ID, settings.GITHUB_OAUTH_CLIENT_SECRET)
    headers = {"Accept": "application/vnd.github+json"}
    payload = {"access_token": access_token}

    try:
        resp = requests.delete(
            _GITHUB_GRANT_DELETE_URL.format(client_id=settings.GITHUB_OAUTH_CLIENT_ID),
            json=payload,
            auth=basic,
            headers=headers,
            timeout=15,
        )
        if resp.status_code == 204:
            return True
        logger.warning(
            f"[GitHubOAuth] Grant deletion returned {resp.status_code}; "
            f"falling back to token revocation."
        )
    except requests.RequestException as exc:
        logger.warning(f"[GitHubOAuth] Grant deletion request failed: {exc}")

    try:
        resp = requests.post(
            _GITHUB_TOKEN_REVOKE_URL.format(client_id=settings.GITHUB_OAUTH_CLIENT_ID),
            json=payload,
            auth=basic,
            headers=headers,
            timeout=15,
        )
        if resp.status_code == 204:
            return True
        logger.warning(f"[GitHubOAuth] Token revocation returned {resp.status_code}")
    except requests.RequestException as exc:
        logger.warning(f"[GitHubOAuth] Token revocation request failed: {exc}")

    return False
