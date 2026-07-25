"""
Query Normalizer
----------------
Normalises free-form user questions into:
  - A clean search query (better for embeddings)
  - Extracted metadata filters (for ChromaDB `where` clauses)
  - Detected intent (for prompt template selection)
"""

import re

# Roman numeral mapping (1-8 for semester I-VIII)
_ARABIC_TO_ROMAN = {
    "1": "I", "2": "II", "3": "III", "4": "IV",
    "5": "V", "6": "VI", "7": "VII", "8": "VIII",
}

_ORDINAL_TO_ROMAN = {
    "first": "I",    "1st": "I",
    "second": "II",  "2nd": "II",
    "third": "III",  "3rd": "III",
    "fourth": "IV",  "4th": "IV",
    "fifth": "V",    "5th": "V",
    "sixth": "VI",   "6th": "VI",
    "seventh": "VII","7th": "VII",
    "eighth": "VIII","8th": "VIII",
}

_ROMAN_NUMERALS = {"I", "II", "III", "IV", "V", "VI", "VII", "VIII"}

# Matches: "semester 1", "sem II", "semester - III", "3rd semester", "first semester"
_SEM_PATTERN = re.compile(
    r"(?:"
    r"(?P<ordinal>first|second|third|fourth|fifth|sixth|seventh|eighth|1st|2nd|3rd|4th|5th|6th|7th|8th)\s+sem(?:ester)?"
    r"|sem(?:ester)?\s*[-]?\s*(?P<roman>[IVXLCDM]{1,8})\b"
    r"|sem(?:ester)?\s+(?P<arabic>[1-8])\b"
    r")",
    re.IGNORECASE,
)

_INTENT_LIST_COURSES = re.compile(
    r"\b(courses?|subjects?|papers?|list|what\s+(are|is)|taught|offered|available)\b",
    re.IGNORECASE,
)
_INTENT_CREDIT_INFO = re.compile(
    r"\b(credits?|total\s+credits?|how many credits?)\b",
    re.IGNORECASE,
)
_INTENT_UNIT = re.compile(
    r"\b(unit\s*[1-6]|module\s*[1-6])\b",
    re.IGNORECASE,
)

_INTENT_OBJECTIVES = re.compile(
    r"\b(objectives?)\b",
    re.IGNORECASE,
)

_INTENT_OUTCOMES = re.compile(
    r"\b(outcomes?)\b",
    re.IGNORECASE,
)
_INTENT_COURSE_DETAIL = re.compile(
    r"\b(syllabus|units?|objectives?|outcomes?|explain|detail|topics?|content)\b",
    re.IGNORECASE,
)


def _extract_semester_roman(text: str):
    m = _SEM_PATTERN.search(text)
    if not m:
        return None
    if m.group("ordinal"):
        return _ORDINAL_TO_ROMAN.get(m.group("ordinal").lower())
    if m.group("roman"):
        r = m.group("roman").upper()
        return r if r in _ROMAN_NUMERALS else None
    if m.group("arabic"):
        return _ARABIC_TO_ROMAN.get(m.group("arabic"))
    return None


def detect_intent(question: str) -> str:
    if _INTENT_CREDIT_INFO.search(question):
        return "credit_info"

    if _INTENT_LIST_COURSES.search(question):
        return "list_courses"

    if _INTENT_UNIT.search(question):
        return "unit"

    if _INTENT_OBJECTIVES.search(question):
        return "objectives"

    if _INTENT_OUTCOMES.search(question):
        return "outcomes"

    if _INTENT_COURSE_DETAIL.search(question):
        return "course_detail"

    return "general"

def normalize_query(question: str) -> dict:
    semester = _extract_semester_roman(question)
    intent = detect_intent(question)

    if semester:
        search_query = _SEM_PATTERN.sub(f"semester {semester} courses", question, count=1)
    else:
        search_query = question

    where = {"semester": semester} if semester else None

    return {
        "search_query": search_query.strip(),
        "where": where,
        "intent": intent,
        "semester": semester,
    }
