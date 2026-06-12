"""03 — Handoff: one agent hands control to another via Command(goto=...).

In the supervisor pattern, routing lives in conditional edges. Handoffs are the
other style: a node itself decides where to go next by returning a Command. This
is how agents pass work to each other ("I can't do this, over to you").

No LLM — pure mechanics.

Run:  python lessons\\05_multi_agent\\03_handoff.py
"""
import sys
from pathlib import Path
from typing import TypedDict

sys.path.append(str(Path(__file__).resolve().parents[2]))

from langgraph.graph import END, START, StateGraph
from langgraph.types import Command


class State(TypedDict):
    question: str
    log: list[str]
    answer: str


def triage(state: State) -> Command:
    """Triage agent: handles greetings itself, hands off anything else."""
    state["log"].append("triage")
    if state["question"].strip().strip("!?.").lower() in {"hi", "hello"}:
        return Command(goto="finish", update={"answer": "Hello!"})
    # Hand off to the worker agent.
    return Command(goto="worker")


def worker(state: State) -> Command:
    """Worker agent: does the real work, then goes to finish."""
    state["log"].append("worker")
    return Command(goto="finish", update={"answer": f"Worked on: {state['question']!r}"})


def finish(state: State) -> State:
    state["log"].append("finish")
    return state


def main() -> None:
    graph = StateGraph(State)
    graph.add_node("triage", triage)
    graph.add_node("worker", worker)
    graph.add_node("finish", finish)
    graph.add_edge(START, "triage")
    graph.add_edge("finish", END)
    app = graph.compile()

    for q in ["Hello", "Summarize the report"]:
        r = app.invoke({"question": q, "log": [], "answer": ""})
        print(f"Q: {q}")
        print(f"   path  : {' -> '.join(r['log'])}")
        print(f"   answer: {r['answer']}")


if __name__ == "__main__":
    main()
