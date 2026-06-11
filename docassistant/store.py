"""Store: wrapper around Qdrant for indexing + retrieval.

Creates the collection on demand, adds chunks, returns a LangChain retriever.
"""
from __future__ import annotations

from langchain_core.documents import Document

from docassistant.config import get_embeddings, settings


def get_vector_store():
    """Return a QdrantVectorStore connected to the configured collection.

    Creates the collection if it does not exist, using the active provider's
    vector size (settings.embedding_dim) and cosine distance.
    """
    from langchain_qdrant import QdrantVectorStore
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams

    client = QdrantClient(url=settings.qdrant_url)

    existing = {c.name for c in client.get_collections().collections}
    if settings.collection not in existing:
        client.create_collection(
            collection_name=settings.collection,
            vectors_config=VectorParams(
                size=settings.embedding_dim, distance=Distance.COSINE
            ),
        )

    return QdrantVectorStore(
        client=client,
        collection_name=settings.collection,
        embedding=get_embeddings(),
    )


def add_documents(chunks: list[Document]) -> int:
    """Index chunks into Qdrant. Returns the number of chunks added."""
    store = get_vector_store()
    store.add_documents(chunks)
    return len(chunks)


def get_retriever(k: int | None = None, **search_kwargs):
    """Retriever for the top-k relevant chunks."""
    k = settings.top_k if k is None else k
    return get_vector_store().as_retriever(search_kwargs={"k": k, **search_kwargs})
