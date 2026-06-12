"""03 — The whole project, wired through MCP.

The document search now lives in a standalone MCP server
(docassistant/mcp_server.py). An agent connects to it as a client and answers
questions using the `search_documents` tool over MCP. This is RAG, but with
retrieval exposed as a reusable MCP service instead of a direct function call.

Index the handbook first:
    python lessons\\03_rag\\02_index.py

Run:  python lessons\\06_mcp\\03_doc_server_agent.py
"""
import asyncio
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ["QDRANT_COLLECTION"] = "lesson03_rag"  # the server inherits this env
sys.path.append(str(Path(__file__).resolve().parents[2]))

from langchain.agents import create_agent
from langchain_mcp_adapters.client import MultiServerMCPClient

from docassistant.config import get_chat_model

# Launch the app's real MCP server as a module.
SERVER_CMD = [sys.executable, "-m", "docassistant.mcp_server"]
PROJECT_ROOT = str(Path(__file__).resolve().parents[2])


async def main() -> None:
    client = MultiServerMCPClient({
        "documents": {
            "command": SERVER_CMD[0],
            "args": SERVER_CMD[1:],
            "transport": "stdio",
            "cwd": PROJECT_ROOT,        # so `-m docassistant.mcp_server` resolves
            "env": dict(os.environ),    # pass QDRANT_COLLECTION etc. to the server
        },
    })
    tools = await client.get_tools()
    print("tools from the document MCP server:", [t.name for t in tools])

    agent = create_agent(get_chat_model(), tools)
    question = "How many vacation days do employees get? Search the documents."
    result = await agent.ainvoke({"messages": [("human", question)]})
    print("\nQ:", question)
    print("A:", result["messages"][-1].content)


if __name__ == "__main__":
    asyncio.run(main())
