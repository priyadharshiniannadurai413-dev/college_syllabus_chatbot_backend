"""
clerk.py
--------
Clerk JWT verification using JWKS.

Verifies Clerk session tokens sent as Bearer tokens in the Authorization header.
Caches the JWKS endpoint response to avoid fetching on every request.
"""

import time
import logging
import threading
from typing import Optional

import requests
from jose import jwt, JWTError
from fastapi import HTTPException, status

from app.core.config import settings

logger = logging.getLogger("uvicorn")

# ── JWKS Cache ────────────────────────────────────────────────────────────────
_jwks_cache: Optional[dict] = None
_jwks_cache_time: float = 0
_jwks_lock = threading.Lock()
JWKS_CACHE_TTL = 3600  # 1 hour


def _fetch_jwks() -> dict:
    """Fetch Clerk's JWKS with caching."""
    global _jwks_cache, _jwks_cache_time

    with _jwks_lock:
        now = time.time()
        if _jwks_cache and (now - _jwks_cache_time) < JWKS_CACHE_TTL:
            return _jwks_cache

        if not settings.CLERK_JWKS_URL:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="CLERK_JWKS_URL is not configured.",
            )

        try:
            resp = requests.get(settings.CLERK_JWKS_URL, timeout=10)
            resp.raise_for_status()
            _jwks_cache = resp.json()
            _jwks_cache_time = now
            logger.info("[Auth] JWKS fetched and cached")
            return _jwks_cache
        except requests.RequestException as exc:
            logger.error(f"[Auth] Failed to fetch JWKS: {exc}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to fetch authentication keys.",
            )


def verify_clerk_token(token: str) -> dict:
    """
    Verify a Clerk session JWT.

    Returns the decoded payload (contains 'sub' user ID, etc.)
    Raises HTTPException on failure.
    """
    try:
        unverified_header = jwt.get_unverified_header(token)
        kid = unverified_header.get("kid")
        if not kid:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token: missing kid.",
            )

        jwks = _fetch_jwks()
        rsa_key = None
        for key in jwks.get("keys", []):
            if key["kid"] == kid:
                rsa_key = {k: key[k] for k in ("kty", "kid", "use", "n", "e") if k in key}
                break

        if not rsa_key:
            logger.warning(f"[Auth] No matching key for kid={kid}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token: no matching key.",
            )

        options = {"verify_aud": False}
        if settings.CLERK_ISSUER:
            payload = jwt.decode(
                token, rsa_key, algorithms=["RS256"],
                issuer=settings.CLERK_ISSUER, options=options,
            )
        else:
            options["verify_iss"] = False
            payload = jwt.decode(
                token, rsa_key, algorithms=["RS256"], options=options,
            )

        return payload

    except HTTPException:
        raise
    except JWTError as exc:
        error_msg = str(exc).lower()
        if "expired" in error_msg:
            logger.warning(f"[Auth] Token expired: {exc}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has expired. Please sign in again.",
            )
        logger.warning(f"[Auth] Token verification failed: {exc}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token.",
        )
