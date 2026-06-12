"""03 — RAG as a graph: the app's docassistant.agents in action.

Same RAG result as stage 3, but now expressed as a graph:
    START -> route -> retrieve -> generate -> END
This uses build_rag_graph() from docassistant/agents.py.

Run 02_index.py from stage 3 first (it indexes the handbook), or upload via the
UI. This lesson points at the stage-3 lesson collection.

Run:  python lessons\\04_langgraph\\03_rag_graph.py
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ["QDRANT_COLLECTION"] = "lesson03_rag"  # reuse stage-3 indexed handbook
sys.path.append(str(Path(__file__).resolve().parents[2]))

from docassistant.agents import build_rag_graph


def main() -> None:
    app = build_rag_graph()

    question = "How many vacation days do I get?"
    result = app.invoke({"question": question})

    print(f"question  : {question}")
    print(f"needs_docs: {result['needs_docs']}")
    print(f"answer    : {result['answer']}")


if __name__ == "__main__":
    main()
