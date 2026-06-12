"""01 — MCP basics: a client connects to a server, lists and calls tools.

MCP separates WHO provides a tool (the server) from WHO uses it (the client).
Here we connect to demo_server.py over stdio (the client launches it as a
subprocess), discover its tools, and call one.

Everything in MCP is async, so we use asyncio.

Run from the project root:
    python lessons\\06_mcp\\01_explore_mcp.py
"""
import asyncio
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from langchain_mcp_adapters.client import MultiServerMCPClient

SERVER = str(Path(__file__).parent / "demo_server.py")


async def main() -> None:
    # Describe how to reach the server: run it with this venv's python over stdio.
    client = MultiServerMCPClient({
        "demo": {"command": sys.executable, "args": [SERVER], "transport": "stdio"},
    })

    # Discover the tools the server exposes (returned as LangChain tools).
    tools = await client.get_tools()
    print("tools exposed by the server:")
    for t in tools:
        print(f"  - {t.name}: {t.description}")

    # Call one tool through MCP.
    add = next(t for t in tools if t.name == "add")
    result = await add.ainvoke({"a": 23, "b": 19})
    print(f"\ncalled add(23, 19) via MCP -> {result}")


if __name__ == "__main__":
    asyncio.run(main())
