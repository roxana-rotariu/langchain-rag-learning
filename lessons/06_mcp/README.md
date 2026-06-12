# Stage 6 — MCP (Model Context Protocol)

MCP standardizes how tools and data are exposed to LLM apps. Instead of hard-wiring
a tool into one agent, you run a **server** that exposes it, and any **client**
(your agent, Claude Desktop, another app) can use it. It decouples the tool from
the agent.

Run from the **project root**, in order:

```powershell
python lessons\06_mcp\01_explore_mcp.py        # client lists + calls server tools
python lessons\06_mcp\02_mcp_in_langgraph.py   # an agent uses MCP tools
python lessons\06_mcp\03_doc_server_agent.py   # agent uses document search via MCP
```

`02`/`03` use the LLM. `03` needs the handbook indexed
(`python lessons\03_rag\02_index.py`). Everything runs locally over **stdio**:
the client launches the server as a subprocess — no network, no keys.

## What you learn and how it maps to the app

| File | Concept | App link |
|---|---|---|
| `demo_server.py` | a minimal MCP server (add, word_count) | how to write a server |
| `01_explore_mcp.py` | client connects, lists, calls tools | server ↔ client |
| `02_mcp_in_langgraph.py` | MCP tools inside a LangGraph agent | `load`/`get_tools` |
| `03_doc_server_agent.py` | retrieval exposed as an MCP service | `docassistant/mcp_server.py` |

## Key ideas

- **Server** exposes tools/resources (`FastMCP` + `@mcp.tool()`).
- **Client** discovers and calls them (`MultiServerMCPClient.get_tools()`).
- **stdio transport**: the client spawns the server as a subprocess and talks
  over stdin/stdout — great for local tools.
- `langchain-mcp-adapters` turns MCP tools into LangChain tools, so any
  LangGraph agent can use them.

## Bonus — use the server from Claude Desktop

`docassistant/mcp_server.py` is a standard MCP server, so other MCP clients can
use it too. In Claude Desktop's MCP config you'd add something like:

```json
{
  "mcpServers": {
    "document-assistant": {
      "command": "C:\\Users\\roxana.curcan\\OsfProjects\\AILearning\\.venv\\Scripts\\python.exe",
      "args": ["-m", "docassistant.mcp_server"],
      "cwd": "C:\\Users\\roxana.curcan\\OsfProjects\\AILearning"
    }
  }
}
```

Then Claude Desktop can search your indexed documents directly. (Optional —
that's the point of MCP: one server, many clients.)

## Exercise

Add a second tool to `docassistant/mcp_server.py` — e.g. `list_sources()` that
returns the distinct document names in the collection. Re-run `03` and ask
"What documents do you have access to?".
