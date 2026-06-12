# Stage 5 — Multi-agent systems

One agent handles everything; a **multi-agent** system splits work across
specialists coordinated by a **supervisor**. The supervisor decides who handles
each task (routing) and agents can hand off to each other.

Run from the **project root**, in order:

```powershell
python lessons\05_multi_agent\01_two_agents.py        # two specialists, in isolation
python lessons\05_multi_agent\02_supervisor.py        # supervisor pattern (mechanics)
python lessons\05_multi_agent\03_handoff.py           # agent -> agent via Command(goto=)
python lessons\05_multi_agent\04_doc_plus_math.py     # real supervisor: RAG + math (LLM routing)
python lessons\05_multi_agent\05_when_multiagent.py   # when is multi-agent worth it?
```

`04` uses the LLM + the stage-3 handbook — run `python lessons\03_rag\02_index.py` first.

## What you learn and how it maps to the app

| File | Concept | App link |
|---|---|---|
| `01_two_agents.py` | a specialist = one focused handler | the idea |
| `02_supervisor.py` | supervisor + conditional routing | the pattern |
| `03_handoff.py` | `Command(goto=...)` handoffs | agent-to-agent |
| `04_doc_plus_math.py` | real supervisor: doc vs math | `agents.build_supervisor_graph()` |
| `05_when_multiagent.py` | single-agent-with-tools vs multi-agent | design judgement |

## Key ideas

- **Supervisor**: a node that routes each task to the right specialist.
- **Handoff**: a node returns `Command(goto="other_agent")` to pass control.
- **Specialists** can have different prompts/tools/models — that's the main reason
  to go multi-agent.
- **Default to one agent with tools.** Split into multiple agents only when a
  single one gets hard to prompt or juggles unrelated jobs.

## Exercise

Add a third specialist to `build_supervisor_graph` in `docassistant/agents.py` —
e.g. a `translate_agent`. Update the supervisor prompt to route to it, and add it
to the conditional-edge mapping. Test with "Translate 'hello' to Romanian".
