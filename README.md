# AI Document Assistant — learning project

An assistant that **answers questions over your documents** (PDF, Word, Excel, PPT),
with source citations. It is both a real application and a learning path: each lesson
in `lessons/` builds a piece of the `docassistant/` package.

The path: chat LLM → embeddings + Qdrant → **RAG with citations** → LangGraph →
multi-agent → MCP.

## Stack

- **LLM:** local via **Ollama** (`qwen2.5:7b`) — free, no API key. Anthropic Claude
  available as a paid fallback (`provider: anthropic` in `config/settings.yaml`).
- **Embeddings:** local **HuggingFace** (`BAAI/bge-small-en-v1.5`, dim 384). Voyage AI
  with the Anthropic backend.
- **Vector DB:** Qdrant (local via Docker)
- **UI:** Streamlit
- **Orchestration:** LangChain v0.3+ / LangGraph
- **Language:** Python 3.12

Switching backends is one line in `config/settings.yaml` (`provider:`) — no code changes.

## Conventions

See [CLAUDE.md](CLAUDE.md): plan before code, strict layering, **no hardcoded text**
(settings/prompts/UI strings live in `config/*.yaml`), all project content in English.

## Structure

```text
docassistant/        # THE APP (code that grows)
  config.py          # settings + prompt/UI loaders + model factories
  ingest.py          # load PDF/Office + chunking
  store.py           # Qdrant: index + retrieval
  rag.py             # question -> retrieve -> answer with citations
  agents.py          # (stages 4-6) LangGraph / multi-agent / MCP
config/              # settings.yaml, prompts.yaml, ui.yaml (no hardcoded text)
app.py               # Streamlit UI (upload + chat)
lessons/             # the learning path (see docs/ROADMAP.md)
docs/ROADMAP.md      # 6-stage plan
check_setup.py       # environment check
```

## Quick start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

docker compose up -d            # Qdrant at http://localhost:6333

# Local provider (default): install Ollama from https://ollama.com, then:
ollama pull qwen2.5:7b          # ~4.7GB, one-time

python check_setup.py           # all green?
streamlit run app.py            # launch the Document Assistant
```

## Switching to the paid Anthropic backend (optional)

Set `provider: anthropic` in `config/settings.yaml`, then put the keys in `.env`:

- `ANTHROPIC_API_KEY` — <https://console.anthropic.com/>
- `VOYAGE_API_KEY` — <https://dashboard.voyageai.com/>
