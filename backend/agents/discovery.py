"""Discovery Agent — scans the market and returns a raw list of candidate startups."""
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy

from backend.agents.common import Emit, run_agent_streaming
from langchain.agents.middleware import ToolCallLimitMiddleware

from backend.agents.repair import ToolCallRepairMiddleware
from backend.config import get_model, get_settings
from backend.prompts import DISCOVERY_PROMPT, DISCOVERY_TASK  # the agent's job description + its task
from backend.schemas import DiscoveryResult
from backend.tools import web_search


#    how many searches it may use
def search_budget(n_startups: int) -> int:
    """More startups need a few more searches; never more than 10 (each costs ~$0.005)."""
    s = get_settings()
    return min(10, max(s.max_searches_discovery, n_startups // 2 + 2))


def build_discovery_agent(max_startups: int):
    s = get_settings()
    budget = search_budget(max_startups)

    return create_agent(
        model=get_model(), #The Groq AI, created once in config.py
        tools=[web_search], #Only web search. It can’t read pages (that’s the Deep Dive’s job).
        system_prompt=DISCOVERY_PROMPT.format(max_startups=max_startups, max_searches=budget),
        response_format=ToolStrategy(DiscoveryResult),  # works on every tool-calling model

        middleware=[
            ToolCallRepairMiddleware(agent="Discovery agent"),  # fixes malformed tool calls (gpt-oss quirk)

            # hard cap on paid Linkup searches; extra calls are blocked and the agent must answer
            ToolCallLimitMiddleware(tool_name="web_search", run_limit=budget),
        ], 
        name="discovery_agent",
    )


async def discover_startups(criteria: str, max_startups: int, emit: Emit) -> DiscoveryResult:
    agent = build_discovery_agent(max_startups)

#drop 🔎
    async def on_search(query: str):
        await emit("search", {"agent": "discovery", "query": query,
                              "message": f"🔎 Searching: {query}"})
#Run the loop
    result: DiscoveryResult = await run_agent_streaming(
        agent,
        DISCOVERY_TASK.format(criteria=criteria, max_startups=max_startups),
        on_search=on_search,
        recursion_limit=40,
        schema=DiscoveryResult,
    )
    # de-duplicate by name and enforce the cap
    seen, unique = set(), []
    for s in result.startups:
        key = s.name.strip().lower()
        if key not in seen:
            seen.add(key)
            unique.append(s)
    result.startups = unique[:max_startups]
    return result

""" Clean the list. The AI sometimes lists the same company twice (“Tabby” and “tabby “).
 .strip().lower() makes names comparable, seen remembers which names were already added,
   and [:max_startups] cuts the list to your number in case the AI returned more."""

