"""04 — Tool calling: the model requests a call, YOU run it, then feed the result back.

This is the fundamental loop behind the agents we build in stage 4+. Here the tool
is a calculator; later the tool will be "search the documents".
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool

from docassistant.config import get_chat_model


@tool
def calculator(expression: str) -> str:
    """Evaluate a simple math expression, e.g. '23 * 19 + 4'."""
    allowed = set("0123456789+-*/(). ")
    if not set(expression) <= allowed:
        return "Error: disallowed characters."
    try:
        return str(eval(expression))  # noqa: S307 — demo only, input filtered
    except Exception as e:  # noqa: BLE001
        return f"Error: {e}"


TOOLS = {"calculator": calculator}


def main() -> None:
    model = get_chat_model().bind_tools(list(TOOLS.values()))

    messages = [HumanMessage("What is 23 * 19 + 4? Use the calculator.")]
    ai = model.invoke(messages)
    messages.append(ai)

    for call in ai.tool_calls:
        print(f"-> model requests tool '{call['name']}' with args {call['args']}")
        result = TOOLS[call["name"]].invoke(call["args"])
        messages.append(ToolMessage(content=result, tool_call_id=call["id"]))

    print("\nFinal answer:", model.invoke(messages).content)


if __name__ == "__main__":
    main()
