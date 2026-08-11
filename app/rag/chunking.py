import re

# ── LangChain imports ──────────────────────────────────────────────────────────
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ──────────────────────────────────────────────────────────────────────────────
# Section-header constants used by the curriculum-overview parser (Phase 1)
# ──────────────────────────────────────────────────────────────────────────────
SECTION_HEADERS = [
    "PROGRAMME SPECIFIC OUTCOMES",
    "PROGRAMME OUTCOMES",
    "INDUCTION PROGRAM",
    "SEMESTER - I",
    "SEMESTER - II",
    "SEMESTER - III",
    "SEMESTER - IV",
    "SEMESTER - V",
    "SEMESTER - VI",
    "SEMESTER - VII",
    "SEMESTER - VIII",
    "PROFESSIONAL ELECTIVE VERTICALS",
    "LIST OF PROFESSIONAL ELECTIVE COURSES(PEC)",
    "LIST OF OPEN ELECTIVES",
    "CREDITS SUMMARY"
]

# Match course codes like 22ZHS101, 22LPC302, 22LPEV306, etc.
COURSE_PATTERN = re.compile(r"^\d{2}[A-Z]{3,5}\d{3}\b")

# Matches semester titles in both formats: "SEMESTER I" and "SEMESTER - I"
SEMESTER_HEADER_PATTERN = re.compile(
    r"^SEMESTER\s*(?:-\s*)?(?:[IVXLCDM]+|\d+)$", re.IGNORECASE
)


def _is_section_header(line: str) -> bool:
    """Check if a line is a known section header (curriculum overview)."""
    upper = line.upper()
    return any(header == upper for header in SECTION_HEADERS)


def _is_semester_header(line: str) -> bool:
    """Check if a line is a semester header (in the detailed syllabus section)."""
    return bool(SEMESTER_HEADER_PATTERN.match(line.strip()))


# ──────────────────────────────────────────────────────────────────────────────
# Core syllabus chunker — kept exactly as originally authored
# ──────────────────────────────────────────────────────────────────────────────
def chunk_by_sections(text):

    chunks = []
    current_chunk = []

    lines = text.split("\n")

    # Two-phase parsing:
    # Phase 1 (detailed_syllabus=False): curriculum overview — split on SECTION_HEADERS
    # Phase 2 (detailed_syllabus=True):  detailed syllabus — split on COURSE_PATTERN
    detailed_syllabus = False

    i = 0

    while i < len(lines):

        line = lines[i].strip()

        if not line:
            i += 1
            continue

        # Skip page separators added by the PDF extractor
        if line.startswith("========== PAGE"):
            i += 1
            continue

        # ── Phase 1: Curriculum overview ──────────────────────────────────────
        if not detailed_syllabus:

            # Detect transition into detailed per-course syllabus
            if COURSE_PATTERN.match(line):
                lookahead = "\n".join(lines[i:i + 8]).upper()
                if "OBJECTIVES" in lookahead:
                    if current_chunk:
                        chunks.append("\n".join(current_chunk))
                    detailed_syllabus = True
                    current_chunk = [line]
                    i += 1
                    continue

            # Split on known section headers
            if _is_section_header(line):
                if current_chunk:
                    chunks.append("\n".join(current_chunk))
                current_chunk = [line]
            else:
                current_chunk.append(line)

        # ── Phase 2: Detailed syllabus ────────────────────────────────────────
        else:

            # Each course code starts a new chunk
            if COURSE_PATTERN.match(line):
                if current_chunk:
                    chunks.append("\n".join(current_chunk))
                current_chunk = [line]

            # Semester headers inside the detailed syllabus start a new chunk too
            elif _is_semester_header(line):
                if current_chunk:
                    chunks.append("\n".join(current_chunk))
                current_chunk = [line]

            else:
                current_chunk.append(line)

        i += 1

    if current_chunk:
        chunks.append("\n".join(current_chunk))

    return chunks


# ──────────────────────────────────────────────────────────────────────────────
# LangChain integration
# ──────────────────────────────────────────────────────────────────────────────


def get_langchain_documents(text: str) -> list[Document]:
    chunks = chunk_by_sections(text)

    documents = [
        Document(
            page_content=chunk,
            metadata={
                "source": "syllabus",
                "section": chunk.splitlines()[0].strip() if chunk.strip() else "",
                "chunk_index": i,
            },
        )
        for i, chunk in enumerate(chunks)
        if chunk.strip()
    ]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
    )

    return splitter.split_documents(documents)