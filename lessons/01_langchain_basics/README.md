# Stage 1 — LangChain Fundamentals

Run from the **project root**:

```powershell
python lessons\01_langchain_basics\01_hello_claude.py
python lessons\01_langchain_basics\02_lcel_chain.py
python lessons\01_langchain_basics\03_structured_output.py
python lessons\01_langchain_basics\04_tool_calling.py
```

## What you learn and how it maps to the Document Assistant

| File | Concept | Where it's used in the app |
|---|---|---|
| `01_hello_claude.py` | invoke / stream / messages | the basis of every answer |
| `02_lcel_chain.py` | LCEL `prompt \| model \| parser` | `rag.py` composes exactly like this |
| `03_structured_output.py` | Pydantic output | extracting metadata from documents |
| `04_tool_calling.py` | the tool-calling loop | the foundation of agents (stage 4+) |

## Key ideas

- **LCEL**: compose components with `|`; each one is a `Runnable` (`.invoke/.stream/.batch`).
- **Tool calling**: the model *requests* the call; YOU run the tool and feed the result back.

## Exercise

In `03`, change the Pydantic schema to extract a `DocumentSummary`
(title, 3 key points, language) from text. That's exactly what you'll use to tag
documents at ingest time.
