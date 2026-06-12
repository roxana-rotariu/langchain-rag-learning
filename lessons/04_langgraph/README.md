# Stage 4 — LangGraph

A chain runs fixed steps. A **graph** has explicit state, makes decisions
(conditional edges), can loop, remembers across turns (checkpointing), and can
pause for human input. This is the foundation of modern agents — LangChain's old
`AgentExecutor` is legacy; LangGraph replaced it.

Run from the **project root**, in order:

```powershell
python lessons\04_langgraph\01_state_graph.py        # state, nodes, edges
python lessons\04_langgraph\02_conditional.py        # the graph decides the path
python lessons\04_langgraph\03_rag_graph.py          # RAG as a graph (needs stage-3 index)
python lessons\04_langgraph\04_agent_decides.py      # retrieve only when needed
python lessons\04_langgraph\05_checkpointing.py      # memory across turns
python lessons\04_langgraph\06_human_in_the_loop.py  # pause for approval, resume
```

`03` and `04` reuse the stage-3 handbook collection — run
`python lessons\03_rag\02_index.py` first if you haven't.

## What you learn and how it maps to the app

| File | Concept | App link |
|---|---|---|
| `01_state_graph.py` | state (TypedDict), nodes, edges | the graph skeleton |
| `02_conditional.py` | conditional edges (routing) | how the agent branches |
| `03_rag_graph.py` | RAG as a graph | `agents.build_rag_graph()` |
| `04_agent_decides.py` | skip retrieval for chit-chat | `agents._route_question` |
| `05_checkpointing.py` | memory per thread_id | future: chat with history |
| `06_human_in_the_loop.py` | interrupt + resume | future: approval steps |

## Key ideas

- **State** is a TypedDict that flows through the graph; nodes return updates to it.
- **Conditional edges** let a routing function pick the next node — this is the
  decision-making a plain chain can't do.
- **Checkpointer** (e.g. `MemorySaver`) persists state by `thread_id` → memory.
- **interrupt()** pauses the graph; **`Command(resume=...)`** continues it.

## Exercise

In `docassistant/agents.py`, extend `_route_question` so a question ending in
"?" always retrieves, and add another greeting to the set. Re-run `04` and watch
the routing change.
