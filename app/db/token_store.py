"""
token_store.py
--------------
Encrypted GitHub token storage in MongoDB.

One document per Clerk user in the 'github_tokens' collection:

    {
        "clerk_user_id":   "user_xxx",
        "encrypted_token": "<Fernet-encrypted OAuth access token>",
        "github_login":    "octocat",
        "scopes":          "repo,read:user",
        "connected_at":    datetime,
    }

Tokens are encrypted at rest with Fernet using TOKEN_ENCRYPTION_KEY.
Plaintext tokens are NEVER logged or returned to clients.
"""

import logging
from datetime import datetime, timezone

from cryptography.fernet import Fernet, InvalidToken

from app.core.config import settings
from app.db.mongodb import db_instance

logger = logging.getLogger("uvicorn")

_fernet: Fernet | None = None
_index_ready = False


def _get_fernet() -> Fernet:
    global _fernet
    if _fernet is None:
        if not settings.TOKEN_ENCRYPTION_KEY:
            raise RuntimeError(
                "TOKEN_ENCRYPTION_KEY is not configured — cannot encrypt GitHub tokens."
            )
        _fernet = Fernet(settings.TOKEN_ENCRYPTION_KEY.encode())
    return _fernet


def get_token_collection():
    db_name = getattr(settings, "DB_NAME", None) or "Chatbot"
    return db_instance.client[db_name]["github_tokens"]


async def _ensure_index() -> None:
    global _index_ready
    if _index_ready:
        return
    await get_token_collection().create_index("clerk_user_id", unique=True)
    _index_ready = True


async def save_token(
    clerk_user_id: str,
    access_token: str,
    github_login: str,
    scopes: str,
) -> None:
    """Encrypt and upsert the user's GitHub access token."""
    encrypted = _get_fernet().encrypt(access_token.encode()).decode()
    await _ensure_index()
    await get_token_collection().update_one(
        {"clerk_user_id": clerk_user_id},
        {
            "$set": {
                "clerk_user_id": clerk_user_id,
                "encrypted_token": encrypted,
                "github_login": github_login,
                "scopes": scopes,
                "connected_at": datetime.now(timezone.utc),
            }
        },
        upsert=True,
    )
    logger.info(
        f"[TokenStore] Saved GitHub token for {clerk_user_id} (login: {github_login})"
    )


async def get_decrypted_token(clerk_user_id: str) -> str | None:
    """Return the plaintext access token for a user, or None if not connected."""
    doc = await get_token_collection().find_one({"clerk_user_id": clerk_user_id})
    if not doc:
        return None
    try:
        return _get_fernet().decrypt(doc["encrypted_token"].encode()).decode()
    except (InvalidToken, KeyError):
        logger.error(
            f"[TokenStore] Decryption failed for {clerk_user_id} — "
            f"TOKEN_ENCRYPTION_KEY may have changed. User must reconnect GitHub."
        )
        return None


async def get_github_login(clerk_user_id: str) -> str | None:
    """Return the connected GitHub username for display, or None."""
    doc = await get_token_collection().find_one(
        {"clerk_user_id": clerk_user_id},
        {"github_login": 1},
    )
    return doc.get("github_login") if doc else None


async def delete_token(clerk_user_id: str) -> bool:
    """Remove the user's stored token. Returns True if one was deleted."""
    result = await get_token_collection().delete_one({"clerk_user_id": clerk_user_id})
    if result.deleted_count:
        logger.info(f"[TokenStore] Deleted GitHub token for {clerk_user_id}")
    return result.deleted_count > 0
