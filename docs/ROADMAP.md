# Roadmap — AI Document Assistant

The app lives in `docassistant/`. Each lesson in `lessons/` builds a piece of it.
Check items off as you progress.

---

## Stage 1 — LangChain Fundamentals · `lessons/01_langchain_basics/` ✅ done

- [x] Chat model (ChatOllama): invoke, stream, system/human messages
- [x] LCEL: `prompt | model | parser`, what a `Runnable` is
- [x] Structured output (Pydantic) — used to extract document metadata
- [x] Tool calling: the `@tool` -> `bind_tools` -> run -> result-back loop
- **Builds in the app:** `docassistant/config.py` — understand the factories.
- **Note:** observed that a small local model hallucinates on niche terms (e.g. "LCEL")
  but answers correctly when given context — the motivation for RAG (stage 3).

## Stage 2 — Embeddings + Qdrant · `lessons/02_embeddings_qdrant/` ✅ done

- [x] What embeddings are; cosine similarity
- [x] Qdrant: collection, points, payload, distance
- [x] Insert + search with `langchain-qdrant`; metadata filtering
- **Builds in the app:** `docassistant/store.py` — index + retriever over the `documents` collection.
- **Note:** confirmed semantic search beats keyword search (0.808 similarity with no shared
  words) and metadata filtering restricts the search to a document subset.

## Stage 3 — RAG with citations · `lessons/03_rag/`  ⭐ ✅ done

- [x] Loaders for PDF / Word / Excel / PPT
- [x] Chunking: `RecursiveCharacterTextSplitter`, chunk size & overlap
- [x] Pipeline: load -> split -> embed -> store -> retrieve -> answer
- [x] Citations: number the fragments, model cites [1][2], show the sources
- [x] Streaming answers + cached retriever in the Streamlit UI
- **Builds in the app:** `docassistant/ingest.py` + `docassistant/rag.py` + launch `app.py` (Streamlit).
- **Milestone:** ✅ uploaded the handbook in the UI and got streamed, cited answers end-to-end.

## Stage 4 — LangGraph · `lessons/04_langgraph/` ✅ done

- [x] `StateGraph`: state (TypedDict), nodes, edges, conditional edges
- [x] Checkpointing (memory across turns), human-in-the-loop (interrupt + resume)
- [x] Why LangGraph > AgentExecutor (legacy)
- **Builds in the app:** `docassistant/agents.py` — RAG as a graph that routes
  greetings to a direct answer and real questions through retrieval.
- **Note:** keyword-based routing is a teaching simplification; real systems route
  with the LLM or tool calling.

## Stage 5 — Multi-agent · `lessons/05_multi_agent/`

- [ ] Supervisor pattern; handoffs; hierarchical
- [ ] When multi-agent beats a single agent with tools
- **Builds in the app:** a supervisor that delegates: one searches documents, another summarizes/compares.

## Stage 6 — MCP · `lessons/06_mcp/`

- [ ] Server vs client MCP; tools/resources
- [ ] Write an MCP server that exposes the document-search tool
- [ ] `langchain-mcp-adapters` — consume MCP tools in the graph
- **Builds in the app:** extract retrieval into a reusable MCP server.

---

### How you work

- Run from the project root: `python lessons\01_langchain_basics\01_hello_claude.py`
- Lessons import from `docassistant.*` — see learning code become application code.
- UI: `streamlit run app.py` (works once you have keys + Qdrant + indexed documents).
