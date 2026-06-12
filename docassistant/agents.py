"""Agents — RAG as a LangGraph graph (stage 4) + multi-agent supervisor (stage 5).

A chain always does the same fixed steps. A graph has explicit STATE and can
DECIDE what to do next (conditional edges), loop, persist memory, and pause for
human input. A multi-agent supervisor routes each question to a specialized
sub-agent (here: documents vs math).

Public interface:
    build_rag_graph(checkpointer=None)  -> compiled RAG graph
    graph_answer(question)              -> str
    build_supervisor_graph()            -> compiled supervisor graph
    supervisor_answer(question)         -> dict {route, answer}

Stage 6 (MCP) builds on this.
"""
from __future__ import annotations

from typing import TypedDict

from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool
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


# --------------------------------------------------------------------------- #
# Stage 5 — multi-agent supervisor
# --------------------------------------------------------------------------- #


@tool
def calculator(expression: str) -> str:
    """Evaluate a simple arithmetic expression, e.g. '23 * 19 + 4'."""
    allowed = set("0123456789+-*/(). ")
    if not set(expression) <= allowed:
        return "Error: disallowed characters."
    try:
        return str(eval(expression))  # demo only — input restricted to the allowlist above
    except Exception as e:
        return f"Error: {e}"


class SupervisorState(TypedDict):
    """State for the supervisor graph."""
    question: str
    route: str       # "doc" | "math", set by the supervisor
    answer: str


def _supervisor(state: SupervisorState) -> SupervisorState:
    """Ask the LLM which specialist should handle the question."""
    instruction = (
        "Route the user's question to a specialist. Reply with ONE word only:\n"
        "- 'math' if it is an arithmetic calculation\n"
        "- 'doc' for anything that should be answered from documents\n"
        f"Question: {state['question']}"
    )
    reply = get_chat_model().invoke(instruction).content.lower()
    state["route"] = "math" if "math" in reply else "doc"
    return state


def _doc_agent(state: SupervisorState) -> SupervisorState:
    """Specialist 1: answer from the documents (RAG)."""
    docs = get_retriever().invoke(state["question"])
    prompt = get_prompt("rag_system") + f"\n\nContext:\n{_format_context(docs)}"
    messages = [("system", prompt), ("human", state["question"])]
    state["answer"] = get_chat_model().invoke(messages).content
    return state


def _math_agent(state: SupervisorState) -> SupervisorState:
    """Specialist 2: solve the calculation using the calculator tool."""
    model = get_chat_model().bind_tools([calculator])
    messages = [HumanMessage(state["question"])]
    ai = model.invoke(messages)
    messages.append(ai)
    for call in ai.tool_calls:
        result = calculator.invoke(call["args"])
        messages.append(ToolMessage(content=result, tool_call_id=call["id"]))
    state["answer"] = model.invoke(messages).content if ai.tool_calls else ai.content
    return state


def _pick_specialist(state: SupervisorState) -> str:
    """Conditional-edge function: route to the chosen specialist node."""
    return state["route"]


def build_supervisor_graph(checkpointer=None):
    """Build a supervisor that delegates to a doc agent or a math agent.

    Flow:  START -> supervisor -> (doc_agent | math_agent) -> END
    """
    graph = StateGraph(SupervisorState)
    graph.add_node("supervisor", _supervisor)
    graph.add_node("doc_agent", _doc_agent)
    graph.add_node("math_agent", _math_agent)

    graph.add_edge(START, "supervisor")
    graph.add_conditional_edges("supervisor", _pick_specialist,
                                {"doc": "doc_agent", "math": "math_agent"})
    graph.add_edge("doc_agent", END)
    graph.add_edge("math_agent", END)

    return graph.compile(checkpointer=checkpointer)


def supervisor_answer(question: str) -> dict:
    """One-shot: run the supervisor graph, return {'route': ..., 'answer': ...}."""
    result = build_supervisor_graph().invoke({"question": question})
    return {"route": result["route"], "answer": result["answer"]}
