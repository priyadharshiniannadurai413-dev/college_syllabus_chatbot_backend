import re
from app.rag.embedding import EmbeddingModel
import logging
from pymongo import UpdateOne
from app.db.mongodb import get_vector_collection


# Matches overview semester headers: "SEMESTER - I", "SEMESTER - IV"
_SEM_HEADER_RE = re.compile(r"SEMESTER\s*-\s*([IVXLCDM]+)", re.IGNORECASE)

# Matches detailed syllabus semester headers: "SEMESTER I", "SEMESTER II"
_SEM_DETAIL_RE = re.compile(r"^SEMESTER\s+([IVXLCDM]+)$", re.IGNORECASE)

# Matches course codes: 22ZHS101, 22LPEV306 etc.
_COURSE_CODE_RE = re.compile(r"^(\d{2}[A-Z]{3,5}\d{3})\b")

_ROMAN_VALID = {"I", "II", "III", "IV", "V", "VI", "VII", "VIII"}


def _parse_chunk_metadata(chunk: str, source: str) -> dict:
    """Extract rich metadata from a chunk's content."""
    first_line = chunk.splitlines()[0].strip() if chunk.strip() else ""
    upper = first_line.upper()

    semester = ""
    chunk_type = "general"
    course_code = ""

    # Check for overview semester header (e.g. "SEMESTER - I")
    m = _SEM_HEADER_RE.search(upper)
    if m:
        roman = m.group(1).upper()
        semester = roman if roman in _ROMAN_VALID else ""
        chunk_type = "overview"

    # Check for detailed syllabus semester header (e.g. "SEMESTER I")
    elif _SEM_DETAIL_RE.match(upper):
        m2 = _SEM_DETAIL_RE.match(upper)
        roman = m2.group(1).upper()
        semester = roman if roman in _ROMAN_VALID else ""
        chunk_type = "detailed"

    # Check for course code (detailed per-course chunk)
    elif _COURSE_CODE_RE.match(first_line):
        course_code = _COURSE_CODE_RE.match(first_line).group(1)
        chunk_type = "detailed"

    return {
        "source": source,
        "title": first_line,
        "semester": semester,
        "type": chunk_type,
        "course_code": course_code,
    }

logger = logging.getLogger("uvicorn")


class VectorStore:
    """
    Stores text chunks alongside their embeddings in MongoDB Atlas,
    and performs vector search using Atlas Vector Search ($vectorSearch).
    """

    INDEX_NAME = "vector_index"

    def _get_collection(self):
        return get_vector_collection()

    async def get_existing_ids(self) -> set[str]:
        """Returns all document chunk IDs already present in MongoDB Atlas."""
        try:
            collection = self._get_collection()
            cursor = collection.find({}, {"_id": 1, "id": 1})
            docs = await cursor.to_list(length=50000)
            existing = set()
            for doc in docs:
                if "_id" in doc:
                    existing.add(str(doc["_id"]))
                if "id" in doc:
                    existing.add(str(doc["id"]))
            return existing
        except Exception as e:
            logger.error(f"Failed to fetch existing document IDs from MongoDB: {e}")
            return set()

    async def ensure_vector_index(self):
        """Automatically checks and creates the Atlas Vector Search index if missing."""
        try:
            collection = self._get_collection()
            cursor = collection.list_search_indexes()
            existing_indexes = await cursor.to_list(length=100)
            idx_names = [idx.get("name") for idx in existing_indexes]
            if self.INDEX_NAME not in idx_names:
                from pymongo.operations import SearchIndexModel
                index_model = SearchIndexModel(
                    definition={
                        "fields": [
                            {
                                "type": "vector",
                                "path": "embedding",
                                "numDimensions": 384,
                                "similarity": "cosine"
                            }
                        ]
                    },
                    name=self.INDEX_NAME,
                    type="vectorSearch"
                )
                await collection.create_search_index(model=index_model)
                logger.info(f"✅ Automatically created Atlas Vector Search index '{self.INDEX_NAME}' on MongoDB Atlas.")
        except Exception as e:
            logger.debug(f"Atlas Search index check note: {e}")

    async def add(
        self,
        documents: list[str] = None,
        embeddings: list[list[float]] = None,
        ids: list[str] = None,
        metadatas: list[dict] = None,
        source: str = "unknown",
        **kwargs
    ):
        """Stores or updates a batch of document chunks in MongoDB Atlas."""
        docs = documents if documents is not None else kwargs.get("chunks", [])
        if not docs or not embeddings:
            return

        if ids is None:
            ids = [f"chunk_{i}" for i in range(len(docs))]

        if metadatas is None:
            metadatas = [_parse_chunk_metadata(doc, source) for doc in docs]

        collection = self._get_collection()
        operations = []

        for chunk_id, embedding, doc, meta in zip(ids, embeddings, docs, metadatas):
            doc_body = {
                "_id": chunk_id,
                "id": chunk_id,
                "text": doc,
                "embedding": embedding,
                "source": meta.get("source", source),
                "metadata": meta,
            }
            operations.append(
                UpdateOne({"_id": chunk_id}, {"$set": doc_body}, upsert=True)
            )

        if operations:
            result = await collection.bulk_write(operations)

            print("Inserted:", result.upserted_count)
            print("Modified:", result.modified_count)

            count = await collection.count_documents({})
            print("Total documents:", count)
            print("MongoDB documents:", collection.count_documents({}))
            logger.info(f"MongoDB Vector Store updated: {result.upserted_count} inserted, {result.modified_count} modified.")
            await self.ensure_vector_index()

    async def query(self, query_embedding: list[float], top_k: int = 4) -> list[dict]:
        """Finds the `top_k` chunks most similar to the query embedding using MongoDB Atlas Vector Search."""
        collection = self._get_collection()

        pipeline = [
            {
                "$vectorSearch": {
                    "index": self.INDEX_NAME,
                    "path": "embedding",
                    "queryVector": query_embedding,
                    "numCandidates": max(top_k * 10, 50),
                    "limit": top_k,
                }
            },
            {
                "$project": {
                    "text": 1,
                    "source": 1,
                    "score": {"$meta": "vectorSearchScore"},
                }
            }
        ]

        try:
            cursor = collection.aggregate(pipeline)
            results = await cursor.to_list(length=top_k)
            matches = []
            for doc in results:
                matches.append({
                    "text": doc.get("text", ""),
                    "source": doc.get("source", "unknown"),
                    "score": doc.get("score", 0.0),
                    "distance": 1.0 - doc.get("score", 0.0),
                })
            return matches
        except Exception as e:
            logger.warning(
                f"MongoDB Vector Search query failed or 'vector_index' is not configured yet in Atlas UI: {e}"
            )
            return []

    async def retrieve(self, search_query: str, top_k: int = 4, where: dict = None) -> dict:
        """
        High-level retrieval interface for RAG pipeline.
        Generates query embedding and returns matches in ChromaDB-compatible dictionary format.
        """
        embedding_model = EmbeddingModel()
        query_embedding = await embedding_model.embed_query(search_query)

        matches = await self.query(query_embedding, top_k=top_k)

        documents = [m["text"] for m in matches]
        return {
            "documents": [documents],
            "matches": matches
        }
