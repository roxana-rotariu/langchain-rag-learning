"""05 — When is multi-agent worth it? (single agent with tools vs supervisor)

Multi-agent is not automatically better. This script frames the trade-off and
demonstrates the simpler alternative: ONE agent with multiple tools. Often that's
enough — reach for multi-agent only when specialists need different prompts,
models, or isolated context.

No new app code — this is a concept lesson with a runnable single-agent example.

Run:  python lessons\\05_multi_agent\\05_when_multiagent.py
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from langchain_core.messages import HumanMessage, ToolMessage

from docassistant.agents import calculator
from docassistant.config import get_chat_model

GUIDANCE = """\
When to use ONE agent with tools:
  - tasks share the same context and persona
  - you just need 'sometimes call a tool' (search, calculator, API)
  - simpler to build, debug, and reason about  <-- default choice

When to use MULTI-AGENT (supervisor / handoffs):
  - specialists need DIFFERENT prompts, models, or tools
  - you want isolated context per role (e.g. a 'critic' that can't see drafts)
  - the workflow has clear stages owned by different responsibilities

Rule of thumb: start with one agent + tools. Split into multiple agents only
when a single one becomes hard to prompt or starts doing too many unrelated jobs.
"""


def single_agent_with_tools(question: str) -> str:
    """One agent, one calculator tool — the simpler alternative to a supervisor."""
    model = get_chat_model().bind_tools([calculator])
    messages = [HumanMessage(question)]
    ai = model.invoke(messages)
    messages.append(ai)
    for call in ai.tool_calls:
        messages.append(ToolMessage(
            content=calculator.invoke(call["args"]), tool_call_id=call["id"]))
    return model.invoke(messages).content if ai.tool_calls else ai.content


def main() -> None:
    print(GUIDANCE)
    print("Single-agent-with-tools demo:")
    print("  Q: What is 35 * 3?")
    print("  A:", single_agent_with_tools("What is 35 * 3? Use the calculator."))


if __name__ == "__main__":
    main()
