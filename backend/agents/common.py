"""Helpers shared by the discovery and deep-dive agents."""

import logging
from typing import Any, Awaitable, Callable

from langchain.messages import AIMessage, ToolMessage
from pydantic import BaseModel

from backend.prompts import SALVAGE_PROMPT

log = logging.getLogger(__name__)

Emit = Callable[[str, dict], Awaitable[None]]


async def run_agent_streaming(
    agent,
    user_message: str,
    on_search: Callable[[str], Awaitable[None]] | None = None,
    recursion_limit: int = 40,
    schema: type[BaseModel] | None = None,
) -> Any:
    """Run a LangChain agent, reporting every `web_search` / `fetch_page` tool call as it happens.

    Returns the agent's `structured_response` (the validated Pydantic object).

    Safety net: if the agent stops without a final structured answer (step limit, model error…), the
    research it already gathered (= paid Linkup calls) is turned into the structured answer with one
    extra LLM call, instead of throwing everything away.
    """
    seen: set[str] = set()
    final_state: dict | None = None
    error: Exception | None = None

    try:
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
    except Exception as e:  # noqa: BLE001
        error = e

    if final_state and final_state.get("structured_response") is not None:
        return final_state["structured_response"]

    if schema is not None and final_state:
        salvaged = await _salvage(final_state.get("messages", []), user_message, schema)
        if salvaged is not None:
            log.warning("Agent stopped early (%s); built the answer from its research notes",
                        str(error)[:120] if error else "no structured response")
            return salvaged
    if error:
        raise error
    raise RuntimeError("Agent finished without a structured response")


async def _salvage(messages: list, task: str, schema: type[BaseModel]) -> BaseModel | None:
    from backend.tools import structured_call  # local import avoids a circular import

    notes = []
    for m in messages:
        if isinstance(m, ToolMessage) and m.name in ("web_search", "fetch_page"):
            notes.append(str(m.content)[:2500])
        elif isinstance(m, AIMessage) and isinstance(m.content, str) and m.content.strip():
            notes.append(m.content[:2500])
    if not notes:
        return None
    research = "\n\n---\n\n".join(notes)[-12000:]
    try:
        return await structured_call(schema, [
            ("system", SALVAGE_PROMPT),
            ("user", f"TASK:\n{task}\n\nRESEARCH NOTES:\n{research}"),
        ])
    except Exception as e:  # noqa: BLE001
        log.warning("Salvage failed: %s", str(e)[:200])
        return None
