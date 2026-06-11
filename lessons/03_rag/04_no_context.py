"""04 — Why RAG? Compare a bare LLM vs a RAG answer on the SAME question.

The handbook says vacation = 25 days. A bare model can't know your company's
policy, so it guesses (hallucinates). RAG gives it the relevant chunk, so it
answers correctly and cites the source. This is the whole point of the project
(recall the 'LCEL' hallucination back in stage 1).

Run 02_index.py first, then:
    python lessons\\03_rag\\04_no_context.py
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ["QDRANT_COLLECTION"] = "lesson03_rag"
sys.path.append(str(Path(__file__).resolve().parents[2]))

from docassistant.config import get_chat_model
from docassistant.rag import answer

QUESTION = "How many vacation days do Acme Corp employees get per year?"


def main() -> None:
    print(f"Question: {QUESTION}\n")

    # 1. Bare LLM — no documents, just the model's own (irrelevant) knowledge.
    bare = get_chat_model().invoke(QUESTION).content
    print("=== BARE LLM (no context) ===")
    print(bare, "\n")

    # 2. RAG — same question, but grounded in the indexed handbook.
    rag = answer(QUESTION)
    print("=== RAG (grounded in handbook) ===")
    print(rag["answer"])
    print(f"\n(grounded in {len(rag['sources'])} chunk(s) from the handbook)")


if __name__ == "__main__":
    main()
