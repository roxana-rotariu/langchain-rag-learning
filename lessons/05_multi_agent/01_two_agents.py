"""01 — Two specialized agents (no graph yet).

An "agent" here is just a function that handles ONE kind of task well. A doc
agent answers from documents; a math agent does arithmetic. On their own they
don't know when to run — that's the supervisor's job (next script).

This script calls each directly to show what they do in isolation. No LLM —
we stub the doc agent so it runs without Qdrant.

Run from the project root:
    python lessons\\05_multi_agent\\01_two_agents.py
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from docassistant.agents import calculator


def doc_agent(question: str) -> str:
    """Stub: pretend to answer from documents (real one uses RAG)."""
    return f"[doc agent] I would search the documents for: {question!r}"


def math_agent(question: str, expression: str) -> str:
    """Uses the real calculator tool from docassistant.agents."""
    return f"[math agent] {expression} = {calculator.invoke({'expression': expression})}"


def main() -> None:
    print(doc_agent("How many vacation days do I get?"))
    print(math_agent("What is 23 * 19 + 4?", "23 * 19 + 4"))
    print("\nEach agent is a specialist. Next: a supervisor that picks the right one.")


if __name__ == "__main__":
    main()
