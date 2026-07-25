from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
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




