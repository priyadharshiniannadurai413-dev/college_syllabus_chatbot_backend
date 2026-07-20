def build_prompt(context: str, question: str, intent: str = "general", semester: str = None) -> str:
    """
    Build an intent-aware RAG prompt for the syllabus chatbot.

    intent: 'list_courses' | 'credit_info' | 'course_detail' | 'general'
    semester: Roman numeral string e.g. 'I', 'IV', or None
    """

    sem_label = f"Semester {semester}" if semester else "the requested semester"

    no_info_msg = "I couldn't find this information in the syllabus."

    if intent == "list_courses":
        return f"""You are SyllabusBot, an academic assistant for GCE Bargur ECE Department (2022 CBCS Regulations).

CONTEXT (retrieved from syllabus):
{context}

TASK:
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

ANSWER:"""

    elif intent == "credit_info":
        return f"""You are SyllabusBot, an academic assistant for GCE Bargur ECE Department (2022 CBCS Regulations).

CONTEXT (retrieved from syllabus):
{context}

TASK:
The student is asking about credit information.

Using ONLY the context above, provide a clear summary of credits. If a credits summary table is available, present it formatted as a markdown table. Show per-semester totals and category-wise breakdown where available.

- Do NOT invent numbers.
- If credit data is not in the context, say: "{no_info_msg}"

QUESTION: {question}

ANSWER:"""

    elif intent == "course_detail":
        return f"""You are SyllabusBot, an academic assistant for GCE Bargur ECE Department (2022 CBCS Regulations).

CONTEXT (retrieved from syllabus):
{context}

TASK:
The student wants detailed information about a specific course or topic.

Using ONLY the context above, provide a structured answer with:
1. **Course Code & Title** (if applicable)
2. **Course Objectives** — list each objective
3. **Unit-wise Topics** — list units with their topics
4. **Course Outcomes** (if available)

- Be concise but complete. Do not skip objectives or units.
- If the requested course is not in the context, say: "{no_info_msg}"

QUESTION: {question}

ANSWER:"""

    else:  # general
        return f"""You are SyllabusBot, an academic assistant for GCE Bargur ECE Department (2022 CBCS Regulations).

CONTEXT (retrieved from syllabus):
{context}

TASK:
Answer the student's question using ONLY the information provided in the context above.

Rules:
- If the answer spans multiple sections, combine them into one complete response.
- Format your answer clearly using bullet points or numbered lists where helpful.
- Do NOT make up course codes, credit values, or unit topics.
- If the answer is not in the context, say: "{no_info_msg}"

QUESTION: {question}

ANSWER:"""