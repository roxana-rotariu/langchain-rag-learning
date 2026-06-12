"""02 — Use MCP tools inside a LangGraph agent.

The payoff: tools from an MCP server become normal LangChain tools, so an agent
can call them. The LLM decides when to use 'add'/'word_count'; the calls go over
MCP to demo_server.py. We use create_react_agent (a prebuilt tool-using agent).

Run:  python lessons\\06_mcp\\02_mcp_in_langgraph.py
"""
import asyncio
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from langchain.agents import create_agent
from langchain_mcp_adapters.client import MultiServerMCPClient

from docassistant.config import get_chat_model

SERVER = str(Path(__file__).parent / "demo_server.py")


async def main() -> None:
    client = MultiServerMCPClient({
        "demo": {"command": sys.executable, "args": [SERVER], "transport": "stdio"},
    })
    tools = await client.get_tools()

    # Build an agent that can use the MCP tools. The LLM picks which to call.
    agent = create_agent(get_chat_model(), tools)

    question = "Use the tools: what is 123 + 877, and how many words are in 'the quick brown fox'?"
    result = await agent.ainvoke({"messages": [("human", question)]})
    print("Q:", question)
    print("A:", result["messages"][-1].content)


if __name__ == "__main__":
    asyncio.run(main())
