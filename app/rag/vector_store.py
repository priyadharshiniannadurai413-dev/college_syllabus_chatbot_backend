import re
import chromadb
from app.rag.embedding import EmbeddingModel

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


class VectorStore:

    def __init__(self):
        self.client = chromadb.PersistentClient(path="./chroma_db")
        self.collection = self.client.get_or_create_collection(name="documents")
        print("Total Documents:", self.collection.count())

    def add(self, chunks, embeddings, source):
        ids = [f"{source}_chunk_{i}" for i in range(len(chunks))]
        metadatas = [_parse_chunk_metadata(chunk, source) for chunk in chunks]

        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=chunks,
            metadatas=metadatas,
        )

    def retrieve(self, question, n_results=8, where=None):
        self.embedding_model = EmbeddingModel()
        query_embedding = self.embedding_model.generate_embeddings([question])[0]

        query_kwargs = {
            "query_embeddings": [query_embedding],
            "n_results": n_results,
        }
        if where:
            query_kwargs["where"] = where

        results = self.collection.query(**query_kwargs)
        return results