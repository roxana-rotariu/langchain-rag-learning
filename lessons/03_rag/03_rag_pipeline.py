"""03 — RAG: question -> retrieve relevant chunks -> answer WITH citations.

This is the heart of the Document Assistant. It calls docassistant.rag.answer(),
the exact function the Streamlit UI uses. Run 02_index.py first.

Run:  python lessons\\03_rag\\03_rag_pipeline.py
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ["QDRANT_COLLECTION"] = "lesson03_rag"  # same collection 02 indexed into
sys.path.append(str(Path(__file__).resolve().parents[2]))

from docassistant.rag import answer

QUESTIONS = [
    "How many vacation days do I get?",
    "Can I work from home?",
    "Who approves a 250 EUR expense?",
]


def main() -> None:
    for q in QUESTIONS:
        result = answer(q)
        print(f"Q: {q}")
        print(f"A: {result['answer']}\n")
        print("   sources:")
        for i, d in enumerate(result["sources"], start=1):
            preview = d.page_content.replace("\n", " ")[:70]
            print(f"     [{i}] {d.metadata.get('source')} — {preview!r}")
        print("-" * 70)


if __name__ == "__main__":
    main()
