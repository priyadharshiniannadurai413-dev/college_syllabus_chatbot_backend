import re
import asyncio
import logging

from pymongo import MongoClient, UpdateOne
from pymongo.operations import SearchIndexModel
from app.db.mongodb import get_vector_collection
from app.rag.embedding import EmbeddingModel
from app.core.config import settings

from langchain_core.documents import Document
from langchain_mongodb import MongoDBAtlasVectorSearch

logger = logging.getLogger("uvicorn")

_SEM_HEADER_RE = re.compile(r"SEMESTER\s*-\s*([IVXLCDM]+)", re.IGNORECASE)
_SEM_DETAIL_RE = re.compile(r"^SEMESTER\s+([IVXLCDM]+)$", re.IGNORECASE)
_COURSE_CODE_RE = re.compile(r"^(\d{2}[A-Z]{3,5}\d{3})\b")
_ROMAN_VALID = {"I", "II", "III", "IV", "V", "VI", "VII", "VIII"}

def _parse_chunk_metadata(chunk: str, source: str) -> dict:
    """Extract rich metadata from a chunk's content."""
    first_line = chunk.splitlines()[0].strip() if chunk.strip() else ""
    upper = first_line.upper()

    semester, chunk_type, course_code = "", "general", ""

    if m := _SEM_HEADER_RE.search(upper):
        roman = m.group(1).upper()
        semester = roman if roman in _ROMAN_VALID else ""
        chunk_type = "overview"
    elif m := _SEM_DETAIL_RE.match(upper):
        roman = m.group(1).upper()
        semester = roman if roman in _ROMAN_VALID else ""
        chunk_type = "detailed"
    elif m := _COURSE_CODE_RE.match(first_line):
        course_code = m.group(1)
        chunk_type = "detailed"

    return {
        "source": source,
        "title": first_line,
        "semester": semester,
        "type": chunk_type,
        "course_code": course_code,
    }

class VectorStore:
    INDEX_NAME = "vector_index"
    COLLECTION_NAME = "vector_documents"

    def __init__(self):
        db_name = getattr(settings, "DB_NAME", "Chatbot")
        sync_collection = MongoClient(settings.MONGODB_URL)[db_name][self.COLLECTION_NAME]
        
        self._lc_store = MongoDBAtlasVectorSearch(
            collection=sync_collection,
            embedding=EmbeddingModel(),
            index_name=self.INDEX_NAME,
            text_key="text",
            embedding_key="embedding",
        )

    @property
    def _collection(self):
        """Returns the async Motor collection (used for admin operations)."""
        return get_vector_collection()

    async def get_existing_ids(self) -> set[str]:
        """Returns all document chunk IDs already present in MongoDB Atlas."""
        try:
            docs = await self._collection.find({}, {"_id": 1, "id": 1}).to_list(length=50000)
            return {str(val) for doc in docs for key, val in doc.items() if key in ("_id", "id")}
        except Exception as e:
            logger.error(f"Failed to fetch existing document IDs from MongoDB: {e}")
            return set()

    async def ensure_vector_index(self):
        """Automatically checks and creates the Atlas Vector Search index if missing."""
        try:
            existing_indexes = await self._collection.list_search_indexes().to_list(length=100)
            if self.INDEX_NAME not in [idx.get("name") for idx in existing_indexes]:
                await self._collection.create_search_index(SearchIndexModel(
                    definition={"fields": [{"type": "vector", "path": "embedding", "numDimensions": 384, "similarity": "cosine"}]},
                    name=self.INDEX_NAME,
                    type="vectorSearch"
                ))
                logger.info(f"✅ Automatically created Atlas Vector Search index '{self.INDEX_NAME}' on MongoDB Atlas.")
        except Exception as e:
            logger.debug(f"Atlas Search index check note: {e}")

    async def add_documents(self, documents: list[Document], source: str = "unknown"):
        if not documents:
            return

        enriched = [
            Document(page_content=doc.page_content, metadata={**doc.metadata, **_parse_chunk_metadata(doc.page_content, source)})
            for doc in documents
        ]

        await asyncio.get_event_loop().run_in_executor(None, lambda: self._lc_store.add_documents(enriched))
        await self.ensure_vector_index()
        count = await self._collection.count_documents({})
        logger.info(f"✅ MongoDB Vector Store updated. Inserted {len(enriched)} documents. Total: {count}.")

    async def add(self, documents: list[str] = None, embeddings: list[list[float]] = None, ids: list[str] = None, metadatas: list[dict] = None, source: str = "unknown", **kwargs):
        docs = documents if documents is not None else kwargs.get("chunks", [])
        if not docs:
            return

        metadatas = metadatas or [_parse_chunk_metadata(doc, source) for doc in docs]
        ids = ids or [f"chunk_{i}" for i in range(len(docs))]

        if embeddings:
            operations = [
                UpdateOne(
                    {"_id": chunk_id},
                    {"$set": {"_id": chunk_id, "id": chunk_id, "text": doc, "embedding": emb, "source": meta.get("source", source), "metadata": meta}},
                    upsert=True
                ) for chunk_id, emb, doc, meta in zip(ids, embeddings, docs, metadatas)
            ]
            if operations:
                result = await self._collection.bulk_write(operations)
                logger.info(f"MongoDB Vector Store updated: {result.upserted_count} inserted, {result.modified_count} modified.")
                await self.ensure_vector_index()
        else:
            lc_docs = [Document(page_content=doc, metadata=meta) for doc, meta in zip(docs, metadatas)]
            await self.add_documents(lc_docs, source=source)

    async def similarity_search(self, query: str, top_k: int = 4) -> list[tuple[Document, float]]:
        try:
            return await asyncio.get_event_loop().run_in_executor(None, lambda: self._lc_store.similarity_search_with_score(query, k=top_k))
        except Exception as e:
            logger.warning(f"MongoDB Vector Search failed or 'vector_index' is not configured yet in Atlas UI: {e}")
            return []

    async def retrieve(self, search_query: str, top_k: int = 4, where: dict = None) -> dict:
        results = await self.similarity_search(search_query, top_k=top_k)
        matches = [{
            "text": doc.page_content,
            "source": doc.metadata.get("source", "unknown"),
            "score": float(score),
            "distance": 1.0 - float(score),
            "metadata": doc.metadata,
        } for doc, score in results]

        return {
            "documents": [[m["text"] for m in matches]],
            "matches": matches,
        }
