"""02 — Supervisor pattern (mechanics, no LLM).

A supervisor is a node that decides WHICH specialist handles the task, then a
conditional edge routes there. This is the same conditional-edge mechanic from
stage 4, now used to coordinate agents. Here the routing is a simple keyword
rule so you see the structure clearly; the real one (script 04) uses the LLM.

Run:  python lessons\\05_multi_agent\\02_supervisor.py
"""
import sys
from pathlib import Path
from typing import TypedDict

sys.path.append(str(Path(__file__).resolve().parents[2]))

from langgraph.graph import END, START, StateGraph


class State(TypedDict):
    question: str
    route: str
    answer: str


def supervisor(state: State) -> State:
    """Decide the specialist by a crude keyword check (real one uses the LLM)."""
    q = state["question"].lower()
    is_math = any(c.isdigit() for c in q) and any(op in q for op in "+-*/")
    state["route"] = "math" if is_math else "doc"
    return state


def doc_agent(state: State) -> State:
    state["answer"] = f"[doc] would search documents for {state['question']!r}"
    return state


def math_agent(state: State) -> State:
    state["answer"] = f"[math] would compute {state['question']!r}"
    return state


def main() -> None:
    graph = StateGraph(State)
    graph.add_node("supervisor", supervisor)
    graph.add_node("doc", doc_agent)
    graph.add_node("math", math_agent)
    graph.add_edge(START, "supervisor")
    graph.add_conditional_edges("supervisor", lambda s: s["route"],
                                {"doc": "doc", "math": "math"})
    graph.add_edge("doc", END)
    graph.add_edge("math", END)
    app = graph.compile()

    for q in ["What is the vacation policy?", "Compute 12 * 8 + 3"]:
        r = app.invoke({"question": q, "route": "", "answer": ""})
        print(f"Q: {q}\n   -> routed to '{r['route']}': {r['answer']}")


if __name__ == "__main__":
    main()
