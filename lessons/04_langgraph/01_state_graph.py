"""01 — StateGraph basics: state, nodes, edges.

A graph holds STATE (a TypedDict). NODES are functions that read the state and
return updates. EDGES say which node runs next. The state flows START -> ... -> END.

No LLM here — just the mechanics, so you see clearly how state moves.

Run from the project root:
    python lessons\\04_langgraph\\01_state_graph.py
"""
import sys
from pathlib import Path
from typing import TypedDict

sys.path.append(str(Path(__file__).resolve().parents[2]))

from langgraph.graph import END, START, StateGraph


class State(TypedDict):
    text: str
    steps: list[str]


def shout(state: State) -> State:
    """Node 1: uppercase the text."""
    state["text"] = state["text"].upper()
    state["steps"].append("shout")
    return state


def exclaim(state: State) -> State:
    """Node 2: add emphasis."""
    state["text"] = state["text"] + "!!!"
    state["steps"].append("exclaim")
    return state


def main() -> None:
    graph = StateGraph(State)
    graph.add_node("shout", shout)
    graph.add_node("exclaim", exclaim)
    graph.add_edge(START, "shout")      # entry
    graph.add_edge("shout", "exclaim")  # shout -> exclaim
    graph.add_edge("exclaim", END)      # exit
    app = graph.compile()

    result = app.invoke({"text": "hello langgraph", "steps": []})
    print("final text :", result["text"])
    print("path taken :", " -> ".join(result["steps"]))


if __name__ == "__main__":
    main()
