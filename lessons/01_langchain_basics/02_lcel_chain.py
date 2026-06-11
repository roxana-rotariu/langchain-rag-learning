"""02 — LCEL: prompt | model | parser. The exact pattern used in docassistant/rag.py.
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from docassistant.config import get_chat_model


def main() -> None:
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an AI tutor. Explain briefly, for a programmer."),
        ("human", "Explain the concept: {concept}"),
    ])
    chain = prompt | get_chat_model() | StrOutputParser()

    print("=== invoke ===")
    print(chain.invoke({"concept": "LCEL"}))

    print("\n=== batch ===")
    for out in chain.batch([{"concept": "Runnable"}, {"concept": "retriever"}]):
        print("-", out[:120], "...")


if __name__ == "__main__":
    main()
