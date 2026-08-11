from langchain_google_genai import GoogleGenerativeAIEmbeddings
from app.core.config import settings


class EmbeddingModel:
    def __init__(
        self,
        model_name: str = "models/gemini-embedding-001",
        dimension: int = 384,
    ):
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model=model_name,
            google_api_key=settings.GEMINI_API_KEY,
            output_dimensionality=dimension,
        )

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Convert document chunks into embeddings."""
        if not texts:
            return []
        return await self.embeddings.aembed_documents(texts)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Convert document chunks into embeddings (sync)."""
        if not texts:
            return []
        return self.embeddings.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        """Convert the user's query into an embedding (sync)."""
        return self.embeddings.embed_query(text)