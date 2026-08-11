from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from app.routes import llm
# from app.api.voice import router as voice_router
from app.db.mongodb import connect_to_mongo, close_mongo_connection
import os

# Ensure outputs directory exists for StaticFiles mounting
os.makedirs("outputs", exist_ok=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await connect_to_mongo()
    yield
    await close_mongo_connection()


app = FastAPI(lifespan=lifespan)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """
    Global handler for JSON body parse errors and Pydantic validation failures.
    Returns a clear, structured error message instead of FastAPI's raw 422 output.
    """
    return JSONResponse(
        status_code=422,
        content={
            "error": "Invalid request body. Ensure you are sending valid JSON with the Content-Type: application/json header.",
            "details": exc.errors(),
            "example": {"user_prompt": "Calculate my CGPA for 8.2, 8.5, 9.0"},
        },
    )


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def landing_page():
    return{
        "message":"this is the landing page"
    }

@app.get("/health")
def health_check():
    return{
        "message":"ok"
    }
app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")
app.include_router(llm.router)
# app.include_router(voice_router)




