# Stage 2 — Embeddings + Qdrant

Run from the **project root**, in order:

```powershell
python lessons\02_embeddings_qdrant\01_embeddings.py
python lessons\02_embeddings_qdrant\02_similarity.py
python lessons\02_embeddings_qdrant\03_qdrant_raw.py
python lessons\02_embeddings_qdrant\04_qdrant_langchain.py
```

First run downloads the embedding model (`bge-small`, ~130MB) — one-time.
Qdrant must be running (`docker compose up -d`).

## What you learn and how it maps to the app

| File | Concept | Where it's used in the app |
|---|---|---|
| `01_embeddings.py` | text -> vector of 384 floats | every chunk gets embedded at ingest |
| `02_similarity.py` | cosine similarity = "closeness in meaning" | the math behind retrieval |
| `03_qdrant_raw.py` | Qdrant low-level: collection/points/payload/search | what `store.py` does under the hood |
| `04_qdrant_langchain.py` | the same via `docassistant.store` + metadata filtering | exactly the app's `store.py` |

## Key ideas

- An **embedding** turns text into a point in high-dimensional space. Similar
  meaning -> nearby points — even with no shared words.
- A **vector DB** (Qdrant) stores those points and finds the nearest ones fast.
- A **point** = id + vector + payload (metadata + the original text).
- **Metadata filtering** narrows the search to a subset (e.g. one document) while
  keeping semantic ranking.

## Exercise

In `02`, add your own anchor + candidates (try Romanian text — `bge-small` is
English-focused, see how scores behave). In `04`, change the filter to
`source='email.txt'` and confirm the invoice chunks disappear from results.
