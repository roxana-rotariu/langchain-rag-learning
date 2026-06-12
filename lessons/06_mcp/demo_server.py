"""A minimal MCP server for the lessons (no Qdrant, so it starts instantly).

It exposes two trivial tools so you can focus on the MCP mechanics (server /
client / tools) without waiting for the embedding model to load.

Clients in 01/02 spawn this file themselves; you don't run it directly. To try
it by hand:  python lessons\\06_mcp\\demo_server.py
"""
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("demo")


@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


@mcp.tool()
def word_count(text: str) -> int:
    """Count the words in a piece of text."""
    return len(text.split())


if __name__ == "__main__":
    mcp.run(transport="stdio")
