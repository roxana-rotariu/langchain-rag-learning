"""01 — Embeddings: turn text into a vector of numbers.

An embedding maps a piece of text to a fixed-length list of floats (384 numbers
with bge-small). Texts with similar meaning get vectors that are close together.

Run from the project root:
    python lessons\\02_embeddings_qdrant\\01_embeddings.py

First run downloads the embedding model (~130MB), one-time.
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")  # quiet a Windows warning
sys.path.append(str(Path(__file__).resolve().parents[2]))

from docassistant.config import get_embeddings, settings


def main() -> None:
    embeddings = get_embeddings()

    text = "A cat sat on the warm windowsill."
    vector = embeddings.embed_query(text)

    print(f"model      : {settings.embedding_model}")
    print(f"text       : {text!r}")
    print(f"vector len : {len(vector)} (matches settings.embedding_dim={settings.embedding_dim})")
    print(f"first 8 dims: {[round(x, 4) for x in vector[:8]]}")
    print("\nThat's it — a piece of text is now a point in 384-dimensional space.")


if __name__ == "__main__":
    main()
