"""05 — Checkpointing: the graph remembers across turns (memory).

A checkpointer saves the state after each run, keyed by a thread_id. Invoke the
same graph again with the same thread_id and it continues from where it left off.
This is how a chatbot remembers the conversation.

We use add_messages so each turn APPENDS to the message history instead of
overwriting it. No LLM needed — we just watch the history grow.

Run:  python lessons\\04_langgraph\\05_checkpointing.py
"""
import sys
from pathlib import Path
from typing import Annotated, TypedDict

sys.path.append(str(Path(__file__).resolve().parents[2]))

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages


class State(TypedDict):
    # add_messages = reducer that appends new messages to the existing list
    messages: Annotated[list, add_messages]


def respond(state: State) -> State:
    """Echo a reply that counts how many messages we've seen so far."""
    count = len(state["messages"])
    return {"messages": [("ai", f"(turn {count // 2 + 1}) I now remember {count} message(s).")]}


def main() -> None:
    graph = StateGraph(State)
    graph.add_node("respond", respond)
    graph.add_edge(START, "respond")
    graph.add_edge("respond", END)

    # The checkpointer persists state; thread_id picks which conversation.
    app = graph.compile(checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": "user-1"}}

    for user_text in ["Hi", "What did I say first?", "And now?"]:
        result = app.invoke({"messages": [("human", user_text)]}, config=config)
        print(f"you : {user_text}")
        print(f"bot : {result['messages'][-1].content}")

    print(f"\nTotal messages remembered on thread 'user-1': {len(result['messages'])}")
    print("A fresh thread_id would start from zero — memory is per-thread.")


if __name__ == "__main__":
    main()
