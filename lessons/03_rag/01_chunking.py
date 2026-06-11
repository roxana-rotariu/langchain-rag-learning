"""01 — Chunking: split a document into overlapping pieces before embedding.

Why chunk? Embeddings work best on small, focused passages. A whole document is
too coarse (one vector for many topics) and may exceed limits. We split into
chunks with a little OVERLAP so a sentence cut at a boundary still appears whole
in a neighbouring chunk.

Run from the project root:
    python lessons\\03_rag\\01_chunking.py
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
sys.path.append(str(Path(__file__).resolve().parents[2]))

from docassistant.config import settings
from docassistant.ingest import load_and_split, load_file

DOC = Path(__file__).parent / "sample_docs" / "handbook.md"


def main() -> None:
    raw = load_file(DOC)
    chunks = load_and_split(DOC)

    print(f"settings: chunk_size={settings.chunk_size}, overlap={settings.chunk_overlap}")
    print(f"loaded {len(raw)} document(s) -> split into {len(chunks)} chunk(s)\n")

    for i, c in enumerate(chunks):
        preview = c.page_content.replace("\n", " ")[:90]
        start = c.metadata.get("start_index")
        print(f"[chunk {i}] {len(c.page_content)} chars | start_index={start}")
        print(f"          {preview!r}\n")


if __name__ == "__main__":
    main()
