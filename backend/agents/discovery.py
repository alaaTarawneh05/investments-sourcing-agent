"""Discovery Agent — scans the market and returns a raw list of candidate startups."""
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy

from backend.agents.common import Emit, run_agent_streaming
from backend.config import get_model
from backend.schemas import DiscoveryResult
from backend.tools import web_search

SYSTEM_PROMPT = """You are a startup-sourcing scout working for a venture-capital fund.
Your job: find real, currently operating STARTUPS that match the investment criteria.

Process:
1. Break the criteria into 3-6 focused web searches (vary wording: sector terms, geography, stage,
   "startup", "raises", "funding round", "seed", "Series A", lists like "top startups in ...").
2. Read the results and collect distinct company names. Prefer companies that clearly satisfy the
   hard constraints (geography, stage, metrics). Skip large corporates, investors, accelerators,
   news outlets and duplicates.
3. Return at most {max_startups} startups, best matches first.

Never invent companies — only return names that appear in the search results.
"""


def build_discovery_agent(max_startups: int):
    return create_agent(
        model=get_model(),
        tools=[web_search],
        system_prompt=SYSTEM_PROMPT.format(max_startups=max_startups),
        response_format=ToolStrategy(DiscoveryResult),  # works on every tool-calling model
        name="discovery_agent",
    )


async def discover_startups(criteria: str, max_startups: int, emit: Emit) -> DiscoveryResult:
    agent = build_discovery_agent(max_startups)

    async def on_search(query: str):
        await emit("search", {"agent": "discovery", "query": query,
                              "message": f"🔎 Searching: {query}"})

    result: DiscoveryResult = await run_agent_streaming(
        agent,
        f"Investment criteria:\n{criteria}\n\nFind up to {max_startups} matching startups.",
        on_search=on_search,
        recursion_limit=30,
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
