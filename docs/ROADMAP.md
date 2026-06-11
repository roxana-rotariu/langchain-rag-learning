# Roadmap — AI Document Assistant

The app lives in `docassistant/`. Each lesson in `lessons/` builds a piece of it.
Check items off as you progress.

---

## Stage 1 — LangChain Fundamentals · `lessons/01_langchain_basics/`

- [ ] Chat model (ChatAnthropic): invoke, stream, system/human messages
- [ ] LCEL: `prompt | model | parser`, what a `Runnable` is
- [ ] Structured output (Pydantic) — used to extract document metadata
- [ ] Tool calling: the `@tool` -> `bind_tools` -> run -> result-back loop
- **Builds in the app:** `docassistant/config.py` — understand the factories.

## Stage 2 — Embeddings + Qdrant · `lessons/02_embeddings_qdrant/`

- [ ] What embeddings are; cosine similarity
- [ ] Qdrant: collection, points, payload, distance
- [ ] Insert + search with `langchain-qdrant`; metadata filtering
- **Builds in the app:** `docassistant/store.py` — index + retriever over the `documents` collection.

## Stage 3 — RAG with citations · `lessons/03_rag/`  ⭐ this is where it gets useful

- [ ] Loaders for PDF / Word / Excel / PPT
- [ ] Chunking: `RecursiveCharacterTextSplitter`, chunk size & overlap
- [ ] Pipeline: load -> split -> embed -> store -> retrieve -> answer
- [ ] Citations: number the fragments, model cites [1][2], show the sources
- [ ] MMR / metadata filtering at retrieval; (optional) hybrid dense+sparse, reranking
- **Builds in the app:** `docassistant/ingest.py` + `docassistant/rag.py` + launch `app.py` (Streamlit).
- **Milestone:** upload a PDF in the UI and get answers with sources. 🎉

## Stage 4 — LangGraph · `lessons/04_langgraph/`

- [ ] `StateGraph`: state (TypedDict), nodes, edges, conditional edges, cycles
- [ ] Checkpointing (memory across turns), human-in-the-loop
- [ ] Why LangGraph > AgentExecutor (legacy)
- **Builds in the app:** `docassistant/agents.py` — RAG as a graph with state + decide retrieve/answer.

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
