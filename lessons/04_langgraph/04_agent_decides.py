"""04 — The graph decides: retrieve only when needed.

A greeting doesn't need document search; a real question does. The route node
sets needs_docs, and the conditional edge skips retrieval for chit-chat. Watch
needs_docs flip between the two questions.

Run:  python lessons\\04_langgraph\\04_agent_decides.py
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ["QDRANT_COLLECTION"] = "lesson03_rag"
sys.path.append(str(Path(__file__).resolve().parents[2]))

from docassistant.agents import build_rag_graph


def main() -> None:
    app = build_rag_graph()

    for question in ["Hello!", "How much vacation do I get?"]:
        result = app.invoke({"question": question})
        route = "retrieved docs" if result["needs_docs"] else "answered directly"
        print(f"Q: {question}")
        print(f"   route : {route}")
        print(f"   answer: {result['answer'][:120]}")
        print("-" * 60)


if __name__ == "__main__":
    main()
