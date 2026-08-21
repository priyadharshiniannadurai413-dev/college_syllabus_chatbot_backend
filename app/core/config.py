from dotenv import load_dotenv
import os

load_dotenv()
class Settings:
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
    MONGODB_URL = os.getenv("MONGODB_URL")
    DB_NAME = os.getenv("DB_NAME", "Chatbot")
    TAVILY_API_KEY=os.getenv("TAVILY_API_KEY")

    GITHUB_PAT = os.getenv("GITHUB_API_KEY")  # legacy — no longer used for MCP

    # Per-user GitHub OAuth App (users connect their own accounts)
    GITHUB_OAUTH_CLIENT_ID = os.getenv("GITHUB_OAUTH_CLIENT_ID")
    GITHUB_OAUTH_CLIENT_SECRET = os.getenv("GITHUB_OAUTH_CLIENT_SECRET")
    GITHUB_OAUTH_REDIRECT_URI = os.getenv("GITHUB_OAUTH_REDIRECT_URI")
    TOKEN_ENCRYPTION_KEY = os.getenv("TOKEN_ENCRYPTION_KEY")

    CLERK_JWKS_URL = os.getenv("CLERK_JWKS_URL")
    CLERK_ISSUER = os.getenv("CLERK_ISSUER")

settings = Settings()

