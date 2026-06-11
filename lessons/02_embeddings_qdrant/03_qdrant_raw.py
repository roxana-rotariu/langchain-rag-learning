"""03 — Qdrant directly (no LangChain): collection, points, payload, search.

This shows what a vector store really is, at the low level:
- a COLLECTION holds POINTS
- each POINT = an id + a vector + a payload (arbitrary metadata, e.g. the text)
- SEARCH = give a query vector, get back the nearest points by cosine distance

Run:  python lessons\\02_embeddings_qdrant\\03_qdrant_raw.py

Uses a throwaway collection so it won't touch the app's data.
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
sys.path.append(str(Path(__file__).resolve().parents[2]))

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from docassistant.config import get_embeddings, settings

COLLECTION = "lesson02_demo"  # separate from the app's "documents_*" collection


def main() -> None:
    embeddings = get_embeddings()
    client = QdrantClient(url=settings.qdrant_url)

    # 1. Create a clean demo collection with the right vector size.
    #    (drop it first if a previous run left it behind)
    if client.collection_exists(COLLECTION):
        client.delete_collection(COLLECTION)
    client.create_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=settings.embedding_dim, distance=Distance.COSINE),
    )

    # 2. Insert a few points: id + vector + payload (the text lives in the payload).
    facts = [
        "Qdrant is an open-source vector database.",
        "Cosine distance measures the angle between vectors.",
        "Python is a popular programming language.",
        "The Eiffel Tower is located in Paris.",
    ]
    vectors = embeddings.embed_documents(facts)
    points = [
        PointStruct(id=i, vector=v, payload={"text": t})
        for i, (t, v) in enumerate(zip(facts, vectors, strict=True))
    ]
    client.upsert(collection_name=COLLECTION, points=points)
    print(f"Inserted {len(points)} points into '{COLLECTION}'.\n")

    # 3. Search: embed a query, ask Qdrant for the closest points.
    query = "What database stores embeddings?"
    hits = client.query_points(
        collection_name=COLLECTION,
        query=embeddings.embed_query(query),
        limit=2,
    ).points

    print(f"query: {query!r}\ntop 2 matches:")
    for h in hits:
        print(f"  score={h.score:.3f}  {h.payload['text']!r}")

    # 4. Clean up the demo collection.
    client.delete_collection(COLLECTION)
    print(f"\nDeleted '{COLLECTION}'. The app's collections are untouched.")


if __name__ == "__main__":
    main()
