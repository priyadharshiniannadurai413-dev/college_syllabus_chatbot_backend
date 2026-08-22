import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parents[2]
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

import asyncio
import logging

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
import litellm

from app.rag.vector_store import get_vector_store
from app.rag.prompt import build_prompt
from app.rag.query_normalizer import normalize_query

logger = logging.getLogger("uvicorn")


# ── Custom LangChain Retriever ────────────────────────────────────────────────

class SyllabusRetriever(BaseRetriever):
    """Wraps VectorStore as a LangChain BaseRetriever."""

    vector_store: VectorStore
    where_filter: dict | None = None
    top_k: int = 4

    class Config:
        arbitrary_types_allowed = True

    async def _aget_relevant_documents(self, query: str) -> list[Document]:
        results = await self.vector_store.retrieve(query, top_k=self.top_k, where=self.where_filter)
        return [
            Document(
                page_content=match["text"],
                metadata={"source": match.get("source", "unknown"), "score": match.get("score", 0.0), **match.get("metadata", {})},
            )
            for match in results.get("matches", [])
        ]

    def _get_relevant_documents(self, query: str) -> list[Document]:
        return asyncio.run(self._aget_relevant_documents(query))


# ── LangChain RAG chain builder ───────────────────────────────────────────────

def _build_chain(retriever: SyllabusRetriever, intent: str, semester: str | None, is_voice: bool):
    prompt_template = ChatPromptTemplate.from_messages([("human", "{prompt}")])

    async def call_llm(messages) -> str:
        response = await litellm.acompletion(
            model="gemini/gemini-2.0-flash",
            messages=[{"role": "user", "content": messages.to_string()}],
        )
        return response.choices[0].message.content

    def build_input(inputs: dict) -> dict:
        context = "\n\n".join(doc.page_content for doc in inputs["context"])
        return {"prompt": build_prompt(context, inputs["question"], intent=intent, semester=semester, is_voice=is_voice)}

    return (
        {"context": retriever | RunnableLambda(lambda docs: docs), "question": RunnablePassthrough()}
        | RunnableLambda(build_input)
        | prompt_template
        | RunnableLambda(call_llm)
        | StrOutputParser()
    )


# ── Public interface ──────────────────────────────────────────────────────────

async def get_rag_response(question: str, is_voice: bool = False) -> str:
    """Full LangChain RAG pipeline: normalize → retrieve → prompt → LLM → answer."""
    normalized = normalize_query(question)
    search_query = normalized["search_query"]
    where_filter = normalized["where"]
    intent = normalized["intent"]
    semester = normalized["semester"]

    logger.info(f"[RAG] Intent: {intent}")
    logger.info(f"[RAG] Search Query: {search_query}")
    logger.info(f"[RAG] Metadata Filter: {where_filter}")

    print(f"\n[RAG] Intent: {intent}")
    print(f"[RAG] Search Query: {search_query}")
    print(f"[RAG] Metadata Filter: {where_filter}")

    retriever = SyllabusRetriever(vector_store=get_vector_store(), where_filter=where_filter, top_k=4)
    retrieved_docs = await retriever._aget_relevant_documents(search_query)

    for i, doc in enumerate(retrieved_docs, start=1):
        print(f"\n========== Chunk {i} ==========")
        print(doc.page_content[:300])

    print("\n========== CONTEXT ==========")
    print("\n\n".join(d.page_content for d in retrieved_docs)[:800])

    answer = await _build_chain(retriever, intent=intent, semester=semester, is_voice=is_voice).ainvoke(question)

    print("\n========== FINAL ANSWER ==========")
    print(answer)

    return answer


async def get_rag_prompt(question: str, is_voice: bool = False) -> str:
    """Backward-compatible wrapper — delegates to get_rag_response()."""
    return await get_rag_response(question, is_voice=is_voice)
