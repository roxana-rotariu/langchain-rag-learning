"""Agents — RAG rebuilt as a LangGraph graph (stage 4).

A chain always does the same fixed steps. A graph has explicit STATE and can
DECIDE what to do next (conditional edges), loop, persist memory, and pause for
human input. Here we wrap the existing rag/store building blocks in a graph.

Public interface:
    build_rag_graph(checkpointer=None) -> compiled graph
    graph_answer(question, ...)        -> str   (convenience one-shot)

Later stages (5 multi-agent, 6 MCP) build on this.
"""
from __future__ import annotations

from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from docassistant.config import get_chat_model, get_prompt
from docassistant.rag import _format_context  # reuse the citation formatting
from docassistant.store import get_retriever


class RagState(TypedDict):
    """State that flows through the graph. Each node reads/updates these keys."""
    question: str
    needs_docs: bool
    context: str
    answer: str


def _route_question(state: RagState) -> RagState:
    """Decide whether the question needs document retrieval at all.

    Trivial chit-chat (greetings) can be answered directly; everything else
    goes through retrieval. Keeps it simple: a tiny keyword heuristic.
    """
    # Strip surrounding punctuation/whitespace so "Hello!" matches "hello".
    q = state["question"].strip().strip("!?.,").lower()
    greetings = {"hi", "hello", "hey", "salut", "buna", "bună", "thanks", "mulțumesc"}
    state["needs_docs"] = q not in greetings
    return state


def _retrieve(state: RagState) -> RagState:
    """Fetch relevant chunks and format them as numbered context."""
    docs = get_retriever().invoke(state["question"])
    state["context"] = _format_context(docs)
    return state


def _generate(state: RagState) -> RagState:
    """Generate the answer. Uses retrieved context if present, else answers plainly."""
    if state.get("context"):
        prompt = get_prompt("rag_system") + f"\n\nContext:\n{state['context']}"
    else:
        prompt = "You are a helpful assistant. Answer briefly."
    messages = [("system", prompt), ("human", state["question"])]
    state["answer"] = get_chat_model().invoke(messages).content
    return state


def _needs_docs(state: RagState) -> str:
    """Conditional-edge function: pick the next node by the routing decision."""
    return "retrieve" if state["needs_docs"] else "generate"


def build_rag_graph(checkpointer=None):
    """Build and compile the RAG graph.

    Flow:  START -> route -> (retrieve -> generate | generate) -> END
    Pass a checkpointer (e.g. MemorySaver) to persist state across turns.
    """
    graph = StateGraph(RagState)
    graph.add_node("route", _route_question)
    graph.add_node("retrieve", _retrieve)
    graph.add_node("generate", _generate)

    graph.add_edge(START, "route")
    graph.add_conditional_edges("route", _needs_docs,
                                {"retrieve": "retrieve", "generate": "generate"})
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)

    return graph.compile(checkpointer=checkpointer)


def graph_answer(question: str) -> str:
    """One-shot convenience: run the graph and return just the answer text."""
    result = build_rag_graph().invoke({"question": question})
    return result["answer"]
