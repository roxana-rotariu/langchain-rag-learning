"""Shared config + factories for the whole app and the lessons.

Single source of truth. Settings come from config/settings.yaml (with env-var
overrides); prompts from config/prompts.yaml; UI strings from config/ui.yaml.
Secrets (API keys) come from .env only.

Two backends, selected by `provider` in settings.yaml (or env LLM_PROVIDER):
  - "local"      -> Ollama (LLM) + HuggingFace (embeddings), free, no API key
  - "anthropic"  -> Claude + Voyage (needs ANTHROPIC_API_KEY + VOYAGE_API_KEY)

    from docassistant.config import settings, get_chat_model, get_embeddings
    from docassistant.config import get_prompt, get_ui_text
"""
from __future__ import annotations

import contextlib
import os
import sys
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import yaml
from dotenv import load_dotenv

# On Windows the console is cp1252 and crashes on non-ASCII prints. Force UTF-8
# once, here, since config is imported everywhere.
if sys.platform == "win32":
    with contextlib.suppress(Exception):
        sys.stdout.reconfigure(encoding="utf-8")

load_dotenv()

CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"


@cache
def _load_yaml(name: str) -> dict:
    """Load and cache a YAML file from the config/ directory."""
    path = CONFIG_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Missing config file: {path}")
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def get_prompt(name: str) -> str:
    """Return a prompt text from config/prompts.yaml by key."""
    prompts = _load_yaml("prompts.yaml")
    if name not in prompts:
        raise KeyError(f"Unknown prompt '{name}'. Available: {sorted(prompts)}")
    return prompts[name]


def get_ui_text(key: str) -> str:
    """Return a UI string from config/ui.yaml by key."""
    ui = _load_yaml("ui.yaml")
    if key not in ui:
        raise KeyError(f"Unknown UI text '{key}'. Available: {sorted(ui)}")
    return ui[key]


@dataclass(frozen=True)
class Settings:
    provider: str            # "local" | "anthropic"
    chat_model: str
    embedding_model: str
    embedding_dim: int
    ollama_url: str          # only meaningful for the local provider
    qdrant_url: str
    collection: str          # already suffixed with the provider name
    chunk_size: int
    chunk_overlap: int
    top_k: int


def _build_settings() -> Settings:
    """Build Settings from settings.yaml, allowing env-var overrides."""
    cfg = _load_yaml("settings.yaml")

    provider = os.getenv("LLM_PROVIDER", cfg["provider"])
    if provider not in cfg:
        raise ValueError(f"Unknown provider '{provider}'. Expected one of: local, anthropic.")

    p = cfg[provider]
    qdrant = cfg.get("qdrant", {})
    chunking = cfg.get("chunking", {})
    retrieval = cfg.get("retrieval", {})

    # Vector size differs per provider, so keep collections separate.
    base_collection = os.getenv("QDRANT_COLLECTION", qdrant["collection"])

    return Settings(
        provider=provider,
        chat_model=os.getenv("CHAT_MODEL", p["chat"]),
        embedding_model=os.getenv("EMBEDDING_MODEL", p["embedding"]),
        embedding_dim=int(p["embedding_dim"]),
        ollama_url=os.getenv("OLLAMA_URL", p.get("ollama_url", "http://localhost:11434")),
        qdrant_url=os.getenv("QDRANT_URL", qdrant["url"]),
        collection=f"{base_collection}_{provider}",
        chunk_size=int(chunking["chunk_size"]),
        chunk_overlap=int(chunking["chunk_overlap"]),
        top_k=int(retrieval["top_k"]),
    )


settings = _build_settings()


def get_chat_model(model: str | None = None, **kwargs):
    """Return a chat model for the active provider (same interface either way)."""
    model = model or settings.chat_model
    if settings.provider == "local":
        from langchain_ollama import ChatOllama

        return ChatOllama(model=model, base_url=settings.ollama_url, **kwargs)

    from langchain_anthropic import ChatAnthropic

    return ChatAnthropic(model=model, **kwargs)


def get_embeddings(model: str | None = None, **kwargs):
    """Return embeddings for the active provider (same interface either way)."""
    model = model or settings.embedding_model
    if settings.provider == "local":
        from langchain_huggingface import HuggingFaceEmbeddings

        return HuggingFaceEmbeddings(model_name=model, **kwargs)

    from langchain_voyageai import VoyageAIEmbeddings

    return VoyageAIEmbeddings(model=model, **kwargs)
