"""MCP server exposing the Document Assistant's retrieval as a reusable tool.

This is real app code: an MCP server that any MCP client (our LangGraph agent,
Claude Desktop, etc.) can connect to and call. It exposes one tool,
`search_documents`, backed by the same Qdrant store the rest of the app uses.

Run it standalone (stdio transport):
    python -m docassistant.mcp_server

Clients usually spawn it themselves (see lessons/06_mcp/).

Windows/stdio note: the stdio transport uses stdout (fd 1) for the JSON-RPC
protocol. Loading the embedding model can write to fd 1 (even from C code) and
corrupt that stream. We therefore (a) warm the model once at startup and
(b) redirect fd 1 -> fd 2 at the OS level while it loads.
"""
from __future__ import annotations

import os
import sys

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("document-assistant")

_store = None  # cached vector store (embedding model loaded once)


def _warm_store():
    """Build the vector store once, loading the embedding model with fd 1
    redirected to fd 2 so model-load output never reaches the protocol stream."""
    global _store
    if _store is None:
        from docassistant.store import get_vector_store

        saved_fd = os.dup(1)
        os.dup2(2, 1)  # point stdout's fd at stderr during the heavy load
        try:
            _store = get_vector_store()
            _store.similarity_search("warmup", k=1)  # force the model to load now
        finally:
            os.dup2(saved_fd, 1)  # restore stdout for the JSON-RPC protocol
            os.close(saved_fd)
    return _store


@mcp.tool()
def search_documents(query: str, k: int = 4) -> str:
    """Search the indexed documents and return the most relevant passages.

    Args:
        query: what to look for.
        k: how many passages to return.
    """
    docs = _warm_store().similarity_search(query, k=k)
    if not docs:
        return "No matching passages found."
    blocks = []
    for i, d in enumerate(docs, start=1):
        source = d.metadata.get("source", "?")
        blocks.append(f"[{i}] ({source}) {d.page_content}")
    return "\n\n".join(blocks)


if __name__ == "__main__":
    _warm_store()  # load the model before the protocol starts using stdout
    print("document-assistant MCP server ready", file=sys.stderr, flush=True)
    mcp.run(transport="stdio")
