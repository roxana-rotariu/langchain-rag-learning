"""01 — Hello Claude: invoke, stream, system/human messages.

Run from the project root:
    python lessons\\01_langchain_basics\\01_hello_claude.py
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))  # project root on path

from langchain_core.messages import HumanMessage, SystemMessage

from docassistant.config import get_chat_model


def main() -> None:
    model = get_chat_model()

    messages = [
        SystemMessage("You are a concise AI tutor. Answer in at most 3 sentences."),
        HumanMessage("What is RAG (Retrieval-Augmented Generation)?"),
    ]
    print("=== invoke ===")
    print(model.invoke(messages).content)

    print("\n=== stream ===")
    for chunk in model.stream([HumanMessage("Give me an analogy for embeddings.")]):
        print(chunk.content, end="", flush=True)
    print()


if __name__ == "__main__":
    main()
