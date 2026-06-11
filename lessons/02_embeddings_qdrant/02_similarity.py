"""02 — Cosine similarity: measure how 'close' two texts are in meaning.

Cosine similarity = how aligned two vectors point (1.0 = identical direction,
0 = unrelated). This is the math behind semantic search: we embed the query and
find the stored vectors with the highest cosine similarity.

Run:  python lessons\\02_embeddings_qdrant\\02_similarity.py
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
sys.path.append(str(Path(__file__).resolve().parents[2]))

from docassistant.config import get_embeddings


def cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity of two vectors, computed by hand (no numpy)."""
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    return dot / (norm_a * norm_b)


def main() -> None:
    embeddings = get_embeddings()

    anchor = "The cat is sleeping on the sofa."
    candidates = [
        "A kitten naps on the couch.",      # same meaning, different words
        "Felines enjoy resting indoors.",   # related topic
        "The stock market fell today.",     # unrelated
    ]

    anchor_vec = embeddings.embed_query(anchor)
    print(f"anchor: {anchor!r}\n")
    # embed_documents embeds a list in one call (more efficient than one-by-one)
    for text, vec in zip(candidates, embeddings.embed_documents(candidates), strict=True):
        print(f"  {cosine(anchor_vec, vec):.3f}  {text!r}")

    print("\nHigher score = closer in meaning. Note: 'kitten/couch' beats 'stock market'")
    print("even though it shares NO words with the anchor — that's semantic search.")


if __name__ == "__main__":
    main()
