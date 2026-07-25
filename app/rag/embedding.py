import asyncio
import litellm
import logging

logger = logging.getLogger("uvicorn")


class EmbeddingModel:
    """
    Turns text into vectors (lists of numbers) so we can compare how
    similar two pieces of text are by comparing their vectors.

    Uses Google's Gemini embedding API (hosted, no local model to download).
    """

    def __init__(self, model_name: str = "gemini/gemini-embedding-001", dimensions: int = 384):
        self.model_name = model_name
        self.dimensions = dimensions

    async def embed_texts(self, texts: list[str], batch_size: int = 10) -> list[list[float]]:
        """Embeds many chunks at once — used during ingestion."""
        embeddings = []

        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            
            # Retry mechanism with exponential backoff
            max_retries = 6
            delay = 2.0
            for attempt in range(max_retries):
                try:
                    response = await litellm.aembedding(
                        model=self.model_name,
                        input=batch,
                        dimensions=self.dimensions,
                        task_type="RETRIEVAL_DOCUMENT",
                    )
                    embeddings.extend(item["embedding"] for item in response.data)
                    break
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise e
                    logger.warning(f"⚠️ Embedding request failed (attempt {attempt + 1}/{max_retries}). Retrying in {delay}s... Error: {e}")
                    await asyncio.sleep(delay)
                    delay *= 2.0

            # Small delay between batches to respect free tier RPM limits
            await asyncio.sleep(1.0)

        return embeddings

    async def embed_query(self, text: str) -> list[float]:
        """Embeds a single piece of text — used for a user's question."""
        max_retries = 6
        delay = 2.0
        for attempt in range(max_retries):
            try:
                response = await litellm.aembedding(
                    model=self.model_name,
                    input=[text],
                    dimensions=self.dimensions,
                    task_type="RETRIEVAL_QUERY",
                )
                return response.data[0]["embedding"]
            except Exception as e:
                if attempt == max_retries - 1:
                    raise e
                logger.warning(f"⚠️ Query embedding request failed. Retrying in {delay}s... Error: {e}")
                await asyncio.sleep(delay)
                delay *= 2.0