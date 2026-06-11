"""02 — Index a document: load -> split -> embed -> store in Qdrant.

This is the full ingest pipeline. We point at a SEPARATE lesson collection
(lesson03_rag_local) so we never touch the app's real data. The override is set
BEFORE importing docassistant, because settings are built at import time.

Run:  python lessons\\03_rag\\02_index.py
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ["QDRANT_COLLECTION"] = "lesson03_rag"  # -> becomes lesson03_rag_local
sys.path.append(str(Path(__file__).resolve().parents[2]))

from qdrant_client import QdrantClient

from docassistant.config import settings
from docassistant.ingest import load_and_split
from docassistant.store import add_documents

DOC = Path(__file__).parent / "sample_docs" / "handbook.md"


def main() -> None:
    # Start clean so re-runs don't pile up duplicate chunks.
    client = QdrantClient(url=settings.qdrant_url)
    if client.collection_exists(settings.collection):
        client.delete_collection(settings.collection)
        print(f"Cleared existing collection '{settings.collection}'.")

    chunks = load_and_split(DOC)
    n = add_documents(chunks)
    print(f"Indexed {n} chunks from {DOC.name} into '{settings.collection}'.")
    print("Now run 03_rag_pipeline.py to ask questions against it.")


if __name__ == "__main__":
    main()
