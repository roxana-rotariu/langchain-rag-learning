"""04 — The real supervisor: doc agent (RAG) + math agent, LLM routing.

Uses build_supervisor_graph() from docassistant/agents.py. The supervisor asks
the LLM whether each question is a 'doc' or 'math' task, then routes to the right
specialist. The doc agent uses RAG; the math agent uses the calculator tool.

Run stage-3 indexing first so the doc agent has something to read:
    python lessons\\03_rag\\02_index.py

Run:  python lessons\\05_multi_agent\\04_doc_plus_math.py
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ["QDRANT_COLLECTION"] = "lesson03_rag"  # reuse the indexed handbook
sys.path.append(str(Path(__file__).resolve().parents[2]))

from docassistant.agents import build_supervisor_graph


def main() -> None:
    app = build_supervisor_graph()

    questions = [
        "How many vacation days do I get?",   # -> doc agent (RAG)
        "What is 144 / 12 + 7?",              # -> math agent (calculator)
    ]
    for q in questions:
        r = app.invoke({"question": q, "route": "", "answer": ""})
        print(f"Q: {q}")
        print(f"   supervisor routed to: {r['route']} agent")
        print(f"   answer: {r['answer']}")
        print("-" * 64)


if __name__ == "__main__":
    main()
