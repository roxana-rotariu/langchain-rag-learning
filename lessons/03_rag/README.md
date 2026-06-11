# Stage 3 — RAG with citations ⭐

The milestone where the Document Assistant becomes useful: ask a question, get an
answer grounded in your documents, with sources.

Run from the **project root**, in order. These use a SEPARATE Qdrant collection
(`lesson03_rag_local`), so they never touch the app's real data.

```powershell
python lessons\03_rag\01_chunking.py        # see how a doc is split
python lessons\03_rag\02_index.py           # load -> split -> embed -> store
python lessons\03_rag\03_rag_pipeline.py    # ask questions, get cited answers
python lessons\03_rag\04_no_context.py      # bare LLM vs RAG (why RAG exists)
```

`02` must run before `03`/`04` (it indexes the handbook). Qdrant must be up.

## What you learn and how it maps to the app

| File | Concept | App module |
|---|---|---|
| `01_chunking.py` | chunk size + overlap, why we split | `ingest.split()` |
| `02_index.py` | the full ingest pipeline | `ingest.load_and_split` + `store.add_documents` |
| `03_rag_pipeline.py` | retrieve -> answer with citations | `rag.answer()` (what the UI calls) |
| `04_no_context.py` | RAG vs bare LLM | the reason the whole project exists |

## Key ideas

- **Chunking**: embeddings work best on small focused passages; overlap keeps
  boundary sentences intact.
- **Grounding**: RAG injects retrieved chunks into the prompt so the model answers
  from your data, not its training memory — and can cite sources.
- **Citations**: we number the fragments so the model can reference `[1]`, `[2]`.

## Milestone — run the real app

After the lessons, launch the Streamlit UI:

```powershell
streamlit run app.py
```

Upload `lessons/03_rag/sample_docs/handbook.md` (or any PDF/Word/Excel/PPT),
click **Index**, then ask questions and expand **Sources**. That's the Document
Assistant working end-to-end.

## Exercise

In `03_rag_pipeline.py`, add a question the handbook does NOT answer (e.g.
"What is the parental leave policy?"). A good RAG system should say it couldn't
find the information — that's the system prompt doing its job (see
`config/prompts.yaml`).
