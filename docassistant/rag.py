"""RAG: question -> retrieve chunks -> answer with citations.

The heart of the Document Assistant. Lesson 03 builds up what lives here.
Prompts come from config/prompts.yaml (no hardcoded text).
"""
from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

from docassistant.config import get_chat_model, get_prompt, settings
from docassistant.store import get_retriever


def _build_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages([
        ("system", get_prompt("rag_system")),
        ("human", get_prompt("rag_human")),
    ])


def _format_context(docs: list[Document]) -> str:
    """Number the fragments so the model can cite [1], [2]..."""
    blocks = []
    for i, d in enumerate(docs, start=1):
        src = d.metadata.get("source", "?")
        page = d.metadata.get("page")
        loc = f"{src}, p.{page}" if page is not None else src
        blocks.append(f"[{i}] ({loc})\n{d.page_content}")
    return "\n\n".join(blocks)


def answer(question: str, k: int | None = None) -> dict:
    """Return {'answer': str, 'sources': list[Document]}."""
    k = k or settings.top_k
    docs = get_retriever(k=k).invoke(question)
    context = _format_context(docs)

    chain = _build_prompt() | get_chat_model()
    response = chain.invoke({"context": context, "question": question})

    return {"answer": response.content, "sources": docs}
