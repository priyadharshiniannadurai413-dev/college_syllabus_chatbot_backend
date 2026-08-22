import re
import asyncio
import logging

from pymongo import MongoClient, UpdateOne
from pymongo.operations import SearchIndexModel
from app.db.mongodb import get_vector_collection
from app.rag.embedding import get_embedding_model
from app.core.config import settings

from langchain_core.documents import Document
from langchain_mongodb import MongoDBAtlasVectorSearch

from app.rag.config import (
    HYBRID_VECTOR_TOP_K,
    HYBRID_KEYWORD_TOP_K,
    HYBRID_FINAL_TOP_K,
    HYBRID_RRF_K,
)

logger = logging.getLogger("uvicorn")

_SEM_HEADER_RE = re.compile(r"SEMESTER\s*-\s*([IVXLCDM]+)", re.IGNORECASE)
_SEM_DETAIL_RE = re.compile(r"^SEMESTER\s+([IVXLCDM]+)$", re.IGNORECASE)
_COURSE_CODE_RE = re.compile(r"^(\d{2}[A-Z]{3,5}\d{3})\b")
_ROMAN_VALID = {"I", "II", "III", "IV", "V", "VI", "VII", "VIII"}


def merge_rrf(
    vector_results: list[tuple[Document, float]],
    keyword_results: list[tuple[Document, float]],
    rrf_k: int = HYBRID_RRF_K,
    final_top_k: int = HYBRID_FINAL_TOP_K,
) -> list[tuple[Document, float]]:
    """
    Reciprocal Rank Fusion — merge two ranked lists into one.

    RRF Score(doc) = SUM( 1 / (k + rank_i) ) for each list where doc appears.
    Duplicate documents (by content hash) are merged automatically.
    """
    doc_scores: dict[str, float] = {}
    doc_map: dict[str, Document] = {}

    def _doc_id(doc: Document) -> str:
        """Stable ID for deduplication: prefer _id in metadata, else first 120 chars."""
        return doc.metadata.get("_id") or doc.page_content[:120]

    for rank, (doc, _score) in enumerate(vector_results):
        did = _doc_id(doc)
        doc_scores[did] = doc_scores.get(did, 0.0) + 1.0 / (rrf_k + rank + 1)
        doc_map[did] = doc

    for rank, (doc, _score) in enumerate(keyword_results):
        did = _doc_id(doc)
        doc_scores[did] = doc_scores.get(did, 0.0) + 1.0 / (rrf_k + rank + 1)
        doc_map[did] = doc

    ranked = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
    return [(doc_map[did], score) for did, score in ranked[:final_top_k]]

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
            embedding=get_embedding_model(),
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

    async def ensure_text_index(self):
        """Create a MongoDB text index on the 'text' field for keyword search."""
        try:
            existing = await self._collection.index_information()
            if "text_index" not in existing:
                await self._collection.create_index(
                    [("text", "text")],
                    name="text_index",
                )
                logger.info("Text index 'text_index' created on 'text' field.")
            else:
                logger.debug("Text index 'text_index' already exists.")
        except Exception as e:
            logger.debug(f"Text index note: {e}")

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

    async def keyword_search(self, query: str, top_k: int = 4) -> list[tuple[Document, float]]:
        """
        MongoDB $text search — performs word-level matching with stemming.
        Requires a text index on the 'text' field (created by ensure_text_index()).
        """
        try:
            cursor = self._collection.find(
                {"$text": {"$search": query}},
                {"score": {"$meta": "textScore"}},
            ).sort(
                [("score", {"$meta": "textScore"})]
            ).limit(top_k)

            docs = await cursor.to_list(length=top_k)
            return [
                (
                    Document(
                        page_content=doc.get("text", ""),
                        metadata={
                            **doc.get("metadata", {}),
                            "_id": str(doc.get("_id", "")),
                        },
                    ),
                    doc.get("score", 0.0),
                )
                for doc in docs
            ]
        except Exception as e:
            logger.warning(f"Keyword search failed (text index may not exist): {e}")
            return []

    async def retrieve(self, search_query: str, top_k: int = 4, where: dict = None) -> dict:
        """
        Hybrid retrieval: runs vector search + keyword search concurrently,
        merges results with Reciprocal Rank Fusion (RRF), and returns top_k.
        """
        vector_results, keyword_results = await asyncio.gather(
            self.similarity_search(search_query, top_k=HYBRID_VECTOR_TOP_K),
            self.keyword_search(search_query, top_k=HYBRID_KEYWORD_TOP_K),
        )

        logger.info(
            f"[Hybrid] Vector: {len(vector_results)}, Keyword: {len(keyword_results)}"
        )

        final_k = min(top_k, HYBRID_FINAL_TOP_K)
        merged = merge_rrf(vector_results, keyword_results, rrf_k=HYBRID_RRF_K, final_top_k=final_k)

        logger.info(f"[Hybrid] Merged: {len(merged)} results")

        matches = [{
            "text": doc.page_content,
            "source": doc.metadata.get("source", "unknown"),
            "score": float(score),
            "distance": 1.0 - float(score),
            "metadata": doc.metadata,
        } for doc, score in merged]

        return {
            "documents": [[m["text"] for m in matches]],
            "matches": matches,
        }


_vector_store_instance: VectorStore | None = None


def get_vector_store() -> VectorStore:
    """Return a shared VectorStore singleton."""
    global _vector_store_instance
    if _vector_store_instance is None:
        _vector_store_instance = VectorStore()
    return _vector_store_instance
