"""Ingest: load documents (PDF / Word / Excel / PPT) and split them into chunks.

Used by the UI (upload) and by lesson 03 (RAG). Returns LangChain Documents with
metadata (source, page/sheet) — essential for citations.
"""
from __future__ import annotations

from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from docassistant.config import settings

# Extension -> loader. Loaders are imported lazily (heavy dependencies).
SUPPORTED = {".pdf", ".docx", ".xlsx", ".xls", ".pptx", ".txt", ".md"}


def load_file(path: str | Path) -> list[Document]:
    """Load a file and return Documents (not yet split)."""
    path = Path(path)
    ext = path.suffix.lower()

    if ext == ".pdf":
        from langchain_community.document_loaders import PyPDFLoader
        docs = PyPDFLoader(str(path)).load()
    elif ext == ".docx":
        from langchain_community.document_loaders import Docx2txtLoader
        docs = Docx2txtLoader(str(path)).load()
    elif ext in {".xlsx", ".xls"}:
        from langchain_community.document_loaders import UnstructuredExcelLoader
        docs = UnstructuredExcelLoader(str(path), mode="elements").load()
    elif ext == ".pptx":
        from langchain_community.document_loaders import UnstructuredPowerPointLoader
        docs = UnstructuredPowerPointLoader(str(path)).load()
    elif ext in {".txt", ".md"}:
        from langchain_community.document_loaders import TextLoader
        docs = TextLoader(str(path), encoding="utf-8").load()
    else:
        raise ValueError(f"Unsupported extension: {ext}. Supported: {sorted(SUPPORTED)}")

    # Ensure 'source' metadata = file name (for citations in the UI)
    for d in docs:
        d.metadata.setdefault("source", path.name)
    return docs


def split(docs: list[Document]) -> list[Document]:
    """Split documents into overlapping chunks (parameters from settings)."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        add_start_index=True,
    )
    return splitter.split_documents(docs)


def load_and_split(path: str | Path) -> list[Document]:
    """Shortcut: load_file + split."""
    return split(load_file(path))
