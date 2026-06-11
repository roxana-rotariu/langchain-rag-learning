"""RAG: question -> retrieve chunks -> answer with citations.

The heart of the Document Assistant. Lesson 03 builds up what lives here.
Prompts come from config/prompts.yaml (no hardcoded text).

Two entry points:
- answer()        -> returns the full answer + sources (used by lessons)
- answer_stream() -> yields answer text chunk-by-chunk, returns sources (used by the UI)

Both accept an optional pre-built retriever so callers (e.g. the cached Streamlit
app) can avoid re-loading the embedding model on every call.
"""
from __future__ import annotations

from collections.abc import Iterator

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


def _retrieve(question: str, k: int | None, retriever) -> list[Document]:
    """Retrieve relevant chunks, using a caller-supplied retriever if given."""
    retriever = retriever or get_retriever(k=settings.top_k if k is None else k)
    return retriever.invoke(question)


def answer(question: str, k: int | None = None, retriever=None) -> dict:
    """Return {'answer': str, 'sources': list[Document]}."""
    docs = _retrieve(question, k, retriever)
    chain = _build_prompt() | get_chat_model()
    response = chain.invoke({"context": _format_context(docs), "question": question})
    return {"answer": response.content, "sources": docs}


def answer_stream(
    question: str, k: int | None = None, retriever=None
) -> tuple[Iterator[str], list[Document]]:
    """Return (token iterator, sources).

    Iterate the first element to stream the answer text; the sources are known
    up front (retrieval happens before generation).
    """
    docs = _retrieve(question, k, retriever)
    chain = _build_prompt() | get_chat_model()
    stream = chain.stream({"context": _format_context(docs), "question": question})
    tokens = (chunk.content for chunk in stream)
    return tokens, docs
