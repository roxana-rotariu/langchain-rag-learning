"""Verify the environment is ready: packages, provider backend, Qdrant.

Run:  python check_setup.py
"""
import sys

# On Windows the console is cp1252 and crashes on non-ASCII prints — force UTF-8.
sys.stdout.reconfigure(encoding="utf-8")

from docassistant.config import settings  # also loads .env

OK = "[ OK ]"
FAIL = "[FAIL]"


def check_packages() -> bool:
    common = ("langchain", "langchain_qdrant", "qdrant_client", "langgraph", "yaml",
              "pypdf", "docx2txt", "openpyxl", "pptx", "streamlit")
    local = ("langchain_ollama", "langchain_huggingface", "sentence_transformers")
    anthropic = ("langchain_anthropic", "langchain_voyageai")
    pkgs = common + (local if settings.provider == "local" else anthropic)

    ok = True
    for pkg in pkgs:
        try:
            __import__(pkg)
            print(f"{OK} import {pkg}")
        except ImportError:
            print(f"{FAIL} import {pkg}  ->  pip install -r requirements.txt")
            ok = False
    return ok


def check_provider() -> bool:
    if settings.provider == "local":
        return _check_ollama()
    return _check_keys()


def _check_ollama() -> bool:
    import urllib.request

    url = settings.ollama_url.rstrip("/") + "/api/tags"
    try:
        with urllib.request.urlopen(url, timeout=3) as resp:
            import json
            tags = json.loads(resp.read())
        names = {m["name"] for m in tags.get("models", [])}
        print(f"{OK} Ollama running at {settings.ollama_url}")
        # model names may carry a :latest tag
        if settings.chat_model in names or f"{settings.chat_model}:latest" in names:
            print(f"{OK} model '{settings.chat_model}' pulled")
            return True
        m = settings.chat_model
        print(f"{FAIL} model '{m}' not pulled  ->  ollama pull {m}")
        return False
    except Exception as e:
        print(f"{FAIL} Ollama unreachable at {settings.ollama_url}: {e}")
        print("       ->  install from https://ollama.com and start it")
        return False


def _check_keys() -> bool:
    import os

    ok = True
    for key in ("ANTHROPIC_API_KEY", "VOYAGE_API_KEY"):
        if os.getenv(key):
            print(f"{OK} {key} set")
        else:
            print(f"{FAIL} {key} missing  ->  add it to .env")
            ok = False
    return ok


def check_qdrant() -> bool:
    try:
        from qdrant_client import QdrantClient

        QdrantClient(url=settings.qdrant_url).get_collections()
        print(f"{OK} Qdrant reachable at {settings.qdrant_url}")
        return True
    except Exception as e:
        print(f"{FAIL} Qdrant unreachable at {settings.qdrant_url}: {e}")
        print("       ->  docker compose up -d")
        return False


if __name__ == "__main__":
    print(f"=== AI Document Assistant — setup check (provider: {settings.provider}) ===\n")
    results = [check_packages(), check_provider(), check_qdrant()]
    print()
    if all(results):
        print("All set. Start lessons/01_langchain_basics/ or `streamlit run app.py` 🎉")
    else:
        print("Fix the [FAIL] lines above, then run again.")
