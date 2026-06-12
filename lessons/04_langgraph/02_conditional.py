"""02 — Conditional edges: the graph decides which path to take.

This is the key difference from a plain chain. A routing function inspects the
state and returns the NAME of the next node. Different inputs -> different paths.

Run:  python lessons\\04_langgraph\\02_conditional.py
"""
import sys
from pathlib import Path
from typing import TypedDict

sys.path.append(str(Path(__file__).resolve().parents[2]))

from langgraph.graph import END, START, StateGraph


class State(TypedDict):
    number: int
    label: str


def classify(state: State) -> State:
    """A no-op node; the real decision happens in the routing function below."""
    return state


def handle_even(state: State) -> State:
    state["label"] = f"{state['number']} is even"
    return state


def handle_odd(state: State) -> State:
    state["label"] = f"{state['number']} is odd"
    return state


def route(state: State) -> str:
    """Routing function: returns the next node's name based on the state."""
    return "even" if state["number"] % 2 == 0 else "odd"


def main() -> None:
    graph = StateGraph(State)
    graph.add_node("classify", classify)
    graph.add_node("even", handle_even)
    graph.add_node("odd", handle_odd)

    graph.add_edge(START, "classify")
    # From 'classify', pick the next node dynamically via route().
    graph.add_conditional_edges("classify", route, {"even": "even", "odd": "odd"})
    graph.add_edge("even", END)
    graph.add_edge("odd", END)
    app = graph.compile()

    for n in (4, 7):
        print(app.invoke({"number": n, "label": ""})["label"])


if __name__ == "__main__":
    main()
