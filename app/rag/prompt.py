# ── LangChain import ───────────────────────────────────────────────────────────
from langchain_core.prompts import ChatPromptTemplate


# ──────────────────────────────────────────────────────────────────────────────
# Intent-specific prompt templates
#
# Each template uses the same variables that the original f-strings used:
#   {context}           — retrieved syllabus chunks
#   {question}          — the student's original question
#   {sem_label}         — "Semester IV" or "the requested semester"
#   {no_info_msg}       — fallback string when context lacks the answer
#   {voice_instruction} — empty string or the VOICE RESPONSE RULES block
#
# All prompt content is kept character-for-character identical to the original.
# ──────────────────────────────────────────────────────────────────────────────

_LIST_COURSES_TEMPLATE = ChatPromptTemplate.from_messages([
    ("human", """\
You are SyllabusBot, an academic assistant for GCE Bargur ECE Department (2022 CBCS Regulations).

CONTEXT (retrieved from syllabus):
{context}

TASK:
{voice_instruction}
The student is asking about the courses offered in {sem_label}.

Using ONLY the context above, list all courses for {sem_label} in this exact format:

| SL.No | Course Code | Course Title | Category | L | T | P | Credits |
|-------|-------------|--------------|----------|---|---|---|---------|
| 1     | XXXXXXX     | ...          | ...      | - | - | - | -       |

- Separate theory and practical sections clearly.
- Show the total credits row at the bottom if available.
- Do NOT invent or guess any values.
- If the context does not contain {sem_label} course data, say: "{no_info_msg}"

QUESTION: {question}

ANSWER:"""),
])

_CREDIT_INFO_TEMPLATE = ChatPromptTemplate.from_messages([
    ("human", """\
You are SyllabusBot, an academic assistant for GCE Bargur ECE Department (2022 CBCS Regulations).

CONTEXT (retrieved from syllabus):
{context}

TASK:
{voice_instruction}
The student is asking about credit information.

Using ONLY the context above, provide a clear summary of credits. If a credits summary table is available, present it formatted as a markdown table. Show per-semester totals and category-wise breakdown where available.

- Do NOT invent numbers.
- If credit data is not in the context, say: "{no_info_msg}"

QUESTION: {question}

ANSWER:"""),
])

_UNIT_TEMPLATE = ChatPromptTemplate.from_messages([
    ("human", """\
You are SyllabusBot, an academic assistant for GCE Bargur ECE Department (2022 CBCS Regulations).

CONTEXT (retrieved from syllabus):
{context}

TASK:
{voice_instruction}

The student is asking only about a specific unit.

Using ONLY the context above:

- Return ONLY the requested unit.
- Do NOT include course objectives.
- Do NOT include course outcomes.
- Do NOT include other units.
- Keep the answer concise.
- If the requested unit is not found, say: "{no_info_msg}"

QUESTION:
{question}

ANSWER:"""),
])

_COURSE_DETAIL_TEMPLATE = ChatPromptTemplate.from_messages([
    ("human", """\
You are SyllabusBot, an academic assistant for GCE Bargur ECE Department (2022 CBCS Regulations).

CONTEXT (retrieved from syllabus):
{context}

TASK:
{voice_instruction}

The student wants detailed information about a specific course or topic.

Using ONLY the context above, provide a structured answer with:
1. Course Code & Title (if applicable)
2. Course Objectives
3. Unit-wise Topics
4. Course Outcomes (if available)

- Be concise but complete.
- If the requested course is not in the context, say: "{no_info_msg}"

QUESTION:
{question}

ANSWER:"""),
])

_GENERAL_TEMPLATE = ChatPromptTemplate.from_messages([
    ("human", """\
You are SyllabusBot, an academic assistant for GCE Bargur ECE Department (2022 CBCS Regulations).

CONTEXT (retrieved from syllabus):
{context}

TASK:
{voice_instruction}
Answer the student's question using ONLY the information provided in the context above.

Rules:
- If the answer spans multiple sections, combine them into one complete response.
- Format your answer clearly using bullet points or numbered lists where helpful.
- Do NOT make up course codes, credit values, or unit topics.
- If the answer is not in the context, say: "{no_info_msg}"

QUESTION: {question}

ANSWER:"""),
])

# Map intent strings to their corresponding ChatPromptTemplate
_INTENT_TEMPLATE_MAP: dict[str, ChatPromptTemplate] = {
    "list_courses":  _LIST_COURSES_TEMPLATE,
    "credit_info":   _CREDIT_INFO_TEMPLATE,
    "unit":          _UNIT_TEMPLATE,
    "course_detail": _COURSE_DETAIL_TEMPLATE,
    "general":       _GENERAL_TEMPLATE,
}


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def build_prompt(
    context: str,
    question: str,
    intent: str = "general",
    semester: str = None,
    is_voice: bool = False,
):
    """
    Build an intent-aware RAG prompt using LangChain's ``ChatPromptTemplate``.

    Parameters
    ----------
    context  : Retrieved syllabus chunks joined as a single string.
    question : The student's original question.
    intent   : One of ``'list_courses'``, ``'credit_info'``, ``'unit'``,
               ``'course_detail'``, ``'general'``.
    semester : Roman numeral string (e.g. ``'I'``, ``'IV'``), or ``None``.
    is_voice : When ``True``, prepends the VOICE RESPONSE RULES block.

    Returns
    -------
    A ``ChatPromptValue`` (LangChain formatted prompt) that can be passed
    directly to an LLM in a LangChain chain, or converted to a string via
    ``.to_string()``.
    """

    # ── Shared template variables ─────────────────────────────────────────────

    sem_label = f"Semester {semester}" if semester else "the requested semester"
    no_info_msg = "I couldn't find this information in the syllabus."

    # Voice instruction block — empty string when not in voice mode so the
    # template renders cleanly without a blank section
    voice_instruction = ""
    if is_voice:
        voice_instruction = """\
VOICE RESPONSE RULES:
- Answer in simple language.
- Keep the answer within 2–3 short sentences.
- Do not use markdown.
- Do not use tables.
- Read naturally as if speaking.
"""

    # ── Select the right template for this intent ─────────────────────────────
    # Fall back to the general template for any unrecognised intent string
    template = _INTENT_TEMPLATE_MAP.get(intent, _GENERAL_TEMPLATE)

    # ── Format and return the ChatPromptValue ─────────────────────────────────
    return template.format_prompt(
        context=context,
        question=question,
        sem_label=sem_label,
        no_info_msg=no_info_msg,
        voice_instruction=voice_instruction,
    )
