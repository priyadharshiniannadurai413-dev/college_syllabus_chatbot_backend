from dotenv import load_dotenv
import os

load_dotenv()
class Settings:
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    MONGODB_URL = os.getenv("MONGODB_URL")
    MONGO_URL = os.getenv("MONGODB_URL") or os.getenv("MONGO_URL")
    DB_NAME = os.getenv("DB_NAME", "Chatbot")

settings = Settings()

