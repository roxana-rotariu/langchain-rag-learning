# CLAUDE.md — AI Document Assistant

Working rules for this project. We follow them strictly — **no sloppy, ad-hoc code.**

## Rule #0 — Language

- **All project content is in English**: code, comments, docstrings, prompts, UI text,
  config, docs, this file.
- The **conversation** with the user is in **Romanian**.
- Exception: the RAG assistant answers its end-user in the question's language — but the
  *prompt text itself* is English.

## Rule #1 — Plan before code

Before any non-trivial code (new module, interface change, new dependency), present a
short plan and wait for OK:

- **What** we build and **which module** it goes in (see layering below).
- The public **interface** (function/class signatures) — not the implementation.
- **Why there**, and which modules it touches.
- Trivial fixes (typo, local rename, obvious bug) do NOT need a plan.

## Rule #2 — No hardcoded text; everything in config

- Settings, prompts, and UI strings live in `config/*.yaml`, never inline in code:
  - `config/settings.yaml` — provider, models, Qdrant, chunking, retrieval
  - `config/prompts.yaml` — prompt texts (`get_prompt(name)`)
  - `config/ui.yaml` — UI strings (`get_ui_text(key)`)
- There can be more YAML files as the app grows; load them through `config.py`.
- **Secrets (API keys) live in `.env` only** — never in YAML (it is version-controlled).
- No magic values (chunk size, dim, model names) scattered in code — all in `settings`.
- No stray `os.getenv` across the codebase — env access goes through `config.py`.

## Rule #3 — Provider abstraction

- The LLM/embeddings backend is chosen by `provider` in `settings.yaml`
  (`local` = Ollama + HuggingFace, free; `anthropic` = Claude + Voyage, paid).
- Code NEVER hardcodes a provider. Always go through `get_chat_model()` /
  `get_embeddings()` — they return the right backend with one interface.
- Qdrant collection name is suffixed with the provider (vector size differs:
  384 local vs 1024 anthropic), so backends don't clash.

## Architecture — layering

Dependencies flow in **one direction**. A lower layer never imports an upper one.

```text
app.py (Streamlit UI)
   └─> docassistant.agents        (stages 4-6: LangGraph / multi-agent / MCP)
          └─> docassistant.rag     (question -> answer with citations)
                 └─> docassistant.store    (Qdrant: index + retrieval)
                 └─> docassistant.ingest   (load + chunk documents)
                        └─> docassistant.config   (settings + factories + text loaders)
```

Layering rules:

- **config** imports no other `docassistant.*` module. It is the base.
- **ingest** and **store** don't know each other; both depend only on `config`.
- **rag** orchestrates `store` (+ optionally `ingest`); it never talks to Qdrant/loaders directly.
- **UI (`app.py`)** and **agents** are the only callers of `rag`. The UI holds no business
  logic (parsing, embeddings, prompts) — it only calls `docassistant` functions.
- **No domain logic in `app.py`.** If it appears, move it into a `docassistant` module.

## Code rules

- **One module = one responsibility.** If a file does two things, split it.
- **Clear interfaces:** type hints everywhere, a short docstring saying *what* it does.
- **Lazy heavy imports** (loaders, qdrant) inside the functions that use them.
- **Explicit errors:** validate input and raise with a clear message; never fail silently.

## Lesson conventions (`lessons/`)

- Each lesson builds a **real piece** of `docassistant/`, not throwaway code.
- Scripts run from the project root and import from `docassistant.*`.
- Each lesson README = what you learn + how it maps to the app module.

## Workflow

- Run from the project root. Qdrant via `docker compose up -d`.
- After package changes: `python check_setup.py` must stay green.
