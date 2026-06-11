"""04 — Same thing through LangChain + the app's own store module.

Compare with 03: LangChain hides the embed -> PointStruct -> upsert dance behind
add_documents()/retriever. We also show metadata filtering (search only within
documents matching a payload field).

Run:  python lessons\\02_embeddings_qdrant\\04_qdrant_langchain.py

This writes into the app's real collection (documents_local). It's just demo data.
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
sys.path.append(str(Path(__file__).resolve().parents[2]))

from langchain_core.documents import Document

from docassistant.store import add_documents, get_retriever


def main() -> None:
    # Documents carry text + metadata. LangChain embeds and stores them for us.
    docs = [
        Document(page_content="The invoice total is 4200 EUR.",
                 metadata={"source": "invoice.pdf"}),
        Document(page_content="Delivery is scheduled for next Monday.",
                 metadata={"source": "email.txt"}),
        Document(page_content="The warranty covers two years.",
                 metadata={"source": "invoice.pdf"}),
    ]
    n = add_documents(docs)
    print(f"Indexed {n} documents via docassistant.store.\n")

    # Plain semantic search: no LangChain PointStruct juggling, just .invoke().
    query = "How much do I owe?"
    print(f"query: {query!r}")
    for d in get_retriever(k=2).invoke(query):
        print(f"  - {d.page_content!r}  (source={d.metadata.get('source')})")

    # Metadata filtering: restrict the search to chunks from a specific source.
    print("\nSame query, filtered to source='invoice.pdf':")
    retriever = get_retriever(k=2, filter={"must": [{"key": "metadata.source",
                                                      "match": {"value": "invoice.pdf"}}]})
    for d in retriever.invoke(query):
        print(f"  - {d.page_content!r}  (source={d.metadata.get('source')})")

    print("\nNote: filtering keeps semantic ranking but only over the allowed subset.")


if __name__ == "__main__":
    main()
