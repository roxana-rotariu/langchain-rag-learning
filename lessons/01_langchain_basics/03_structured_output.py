"""03 — Structured output (Pydantic).

In the Document Assistant we use this to extract metadata from a document
(title, key points, language) at ingest time — see the exercise in the README.
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from pydantic import BaseModel, Field

from docassistant.config import get_chat_model

SAMPLE = """Quarterly report Q2. Revenue grew 12% versus Q1, driven by the Eastern
European market. Gross margin held at 41%. The main risk remains dependency on a
single component supplier."""


class DocumentSummary(BaseModel):
    """Structured summary of a document, used as ingest-time metadata."""
    title: str = Field(description="a short, descriptive title")
    key_points: list[str] = Field(description="2-4 key points")
    language: str = Field(description="the document language, e.g. 'ro', 'en'")


def main() -> None:
    model = get_chat_model().with_structured_output(DocumentSummary)
    summary = model.invoke(f"Summarize the document:\n\n{SAMPLE}")

    print("Title:   ", summary.title)
    print("Language:", summary.language)
    print("Key points:")
    for p in summary.key_points:
        print("  -", p)


if __name__ == "__main__":
    main()
