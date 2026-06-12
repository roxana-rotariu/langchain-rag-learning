"""06 — Human-in-the-loop: pause the graph for approval, then resume.

interrupt() stops the graph mid-run and hands control back to your code. You
inspect what it wants, then resume with Command(resume=...). This is how you add
an approval step (e.g. before a destructive action or sending an email).

Requires a checkpointer (the paused state must be saved somewhere).

Run:  python lessons\\04_langgraph\\06_human_in_the_loop.py
"""
import sys
from pathlib import Path
from typing import TypedDict

sys.path.append(str(Path(__file__).resolve().parents[2]))

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt


class State(TypedDict):
    action: str
    approved: bool
    result: str


def ask_approval(state: State) -> State:
    """Pause and ask a human to approve the action."""
    decision = interrupt({"question": f"Approve action '{state['action']}'?"})
    state["approved"] = decision == "yes"
    return state


def execute(state: State) -> State:
    state["result"] = (
        f"Executed: {state['action']}" if state["approved"]
        else f"Cancelled: {state['action']}"
    )
    return state


def main() -> None:
    graph = StateGraph(State)
    graph.add_node("ask_approval", ask_approval)
    graph.add_node("execute", execute)
    graph.add_edge(START, "ask_approval")
    graph.add_edge("ask_approval", "execute")
    graph.add_edge("execute", END)
    app = graph.compile(checkpointer=MemorySaver())

    config = {"configurable": {"thread_id": "approval-demo"}}

    # 1. Start the graph. It runs until interrupt(), then PAUSES.
    paused = app.invoke({"action": "delete all temp files"}, config=config)
    print("graph paused, asking:", paused["__interrupt__"][0].value["question"])

    # 2. A human answers. Resume the graph, feeding the decision into interrupt().
    decision = "yes"  # imagine this came from a UI button
    print(f"human decides: {decision}")
    final = app.invoke(Command(resume=decision), config=config)

    print("final result :", final["result"])


if __name__ == "__main__":
    main()
