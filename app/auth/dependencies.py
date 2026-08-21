"""
dependencies.py
---------------
FastAPI dependencies for authentication.

Provides get_current_user() which extracts and verifies the Clerk JWT
from the Authorization header.
"""

import logging
from typing import Optional

from fastapi import Request, HTTPException, status

from app.auth.clerk import verify_clerk_token

logger = logging.getLogger("uvicorn")


async def get_current_user(request: Request) -> dict:
    """
    FastAPI dependency that extracts the Bearer token from the
    Authorization header, verifies it against Clerk's JWKS,
    and returns the decoded user payload.

    Returns dict with at least: {"sub": "<clerk_user_id>", ...}
    Raises 401 if token is missing, expired, or invalid.
    """
    auth_header: Optional[str] = request.headers.get("Authorization")

    if not auth_header:
        logger.info("[Auth] No Authorization header provided")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required. Please sign in.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    parts = auth_header.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        logger.info(f"[Auth] Malformed Authorization header: {auth_header[:20]}...")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid Authorization header format. Use: Bearer <token>",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = parts[1]
    payload = verify_clerk_token(token)

    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token payload: missing user ID.",
        )

    return payload
