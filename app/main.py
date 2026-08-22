from contextlib import asynccontextmanager
import os

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.routes import llm
from app.routes import github_auth
from app.db.mongodb import connect_to_mongo, close_mongo_connection



# Ensure outputs directory exists for StaticFiles mounting
os.makedirs("outputs", exist_ok=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application startup and shutdown lifecycle.
    """

    # Startup
    await connect_to_mongo()


    yield

    

    await close_mongo_connection()


app = FastAPI(
    lifespan=lifespan
)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
):
    """
    Global handler for JSON body parse errors and
    Pydantic validation failures.
    """

    return JSONResponse(
        status_code=422,
        content={
            "error": (
                "Invalid request body. Ensure you are sending valid JSON "
                "with the Content-Type: application/json header."
            ),
            "details": exc.errors(),
            "example": {
                "user_prompt": "Calculate my CGPA for 8.2, 8.5, 9.0"
            },
        },
    )


# Allowed frontend origins
# NOTE: localhost and 127.0.0.1 are DIFFERENT origins for CORS — allow both so
# the app works regardless of which URL the dev server prints.
_allowed_origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]


# Add deployed frontend URL from environment variable
# NOTE: rstrip("/") — CORSMiddleware does EXACT origin matching, and browsers
# send the Origin header WITHOUT a trailing slash, so "https://x.vercel.app/"
# would silently never match.
_frontend_url = os.getenv("FRONTEND_URL", "").strip().rstrip("/")

if _frontend_url:
    _allowed_origins.append(_frontend_url)


# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def landing_page():
    return {
        "message": "This is the landing page"
    }


@app.get("/health")
def health_check():
    """
    Health check endpoint.
    """

    return {
        "message": "ok"
    }


# Static files
app.mount(
    "/outputs",
    StaticFiles(directory="outputs"),
    name="outputs",
)


# API routes
app.include_router(llm.router)
app.include_router(github_auth.router)

# app.include_router(voice_router)