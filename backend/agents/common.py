"""Helpers shared by the discovery and deep-dive agents."""
from typing import Any, Awaitable, Callable

from langchain.messages import AIMessage

Emit = Callable[[str, dict], Awaitable[None]]


async def run_agent_streaming(
    agent,
    user_message: str,
    on_search: Callable[[str], Awaitable[None]] | None = None,
    recursion_limit: int = 30,
) -> Any:
    """Run a LangChain agent, reporting every `web_search` / `fetch_page` tool call as it happens.

    Returns the agent's `structured_response` (the validated Pydantic object).
    """
    seen: set[str] = set()
    final_state: dict | None = None

    async for state in agent.astream(
        {"messages": [{"role": "user", "content": user_message}]},
        config={"recursion_limit": recursion_limit},
        stream_mode="values",
    ):
        final_state = state
        if not on_search or not state.get("messages"):
            continue
        last = state["messages"][-1]
        if isinstance(last, AIMessage) and last.tool_calls:
            for tc in last.tool_calls:
                if tc["id"] in seen:
                    continue
                seen.add(tc["id"])
                if tc["name"] == "web_search":
                    await on_search(tc["args"].get("query", ""))
                elif tc["name"] == "fetch_page":
                    await on_search("📄 reading " + tc["args"].get("url", ""))

    if not final_state or final_state.get("structured_response") is None:
        raise RuntimeError("Agent finished without a structured response")
    return final_state["structured_response"]
