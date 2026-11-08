"""Deep Dive Agents — one sub-agent per startup, run concurrently."""
import asyncio
import logging

from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy

from backend.agents.common import Emit, run_agent_streaming
from langchain.agents.middleware import ToolCallLimitMiddleware

from backend.agents.repair import ToolCallRepairMiddleware
from backend.config import get_model, get_settings
from backend.prompts import DEEP_DIVE_PROMPT, DEEP_DIVE_TASK  # the analyst's job description + its task
from backend.schemas import AttributeValue, StartupCandidate, StartupProfile
from backend.tools import fetch_page, web_search

log = logging.getLogger(__name__)

_agent = None


def get_deep_dive_agent():
    global _agent
    if _agent is None:  # stateless, so a single compiled agent can serve all parallel runs
        s = get_settings()

        _agent = create_agent(
            model=get_model(),
            tools=[web_search, fetch_page],
            system_prompt=DEEP_DIVE_PROMPT.format(max_searches=s.max_searches_per_startup,
                                               max_reads=s.max_page_reads_per_startup),
            response_format=ToolStrategy(StartupProfile),  # works on every tool-calling model
            middleware=[
                ToolCallRepairMiddleware(agent="Deep Dive agent"),  # fixes malformed tool calls (gpt-oss quirk)
                # hard caps on paid Linkup calls per startup
                ToolCallLimitMiddleware(tool_name="web_search", run_limit=s.max_searches_per_startup),
                ToolCallLimitMiddleware(tool_name="fetch_page", run_limit=s.max_page_reads_per_startup),
            ],
            name="deep_dive_agent",
        )
    return _agent

#  check and tidy the answer
def _normalize(profile: StartupProfile, startup: StartupCandidate, attributes: list[str]) -> StartupProfile:
    """Guarantee one value per requested attribute, in order (data consistency)."""
    by_name = {a.attribute.strip().lower(): a for a in profile.attributes}
    fixed = []
    for attr in attributes:
        found = by_name.get(attr.strip().lower())
        if found is None:
            if attr.lower() == "company name":
                found = AttributeValue(attribute=attr, value=startup.name)
            elif attr.lower() == "website" and startup.website:
                found = AttributeValue(attribute=attr, value=startup.website, source_url=startup.website)
            else:
                found = AttributeValue(attribute=attr, value="Not found")
        found.attribute = attr
        fixed.append(found)
    profile.attributes = fixed
    profile.company_name = startup.name  # canonical name = the one Discovery found (links report sections)
    return profile



#research one startup 
async def deep_dive_one(startup: StartupCandidate, attributes: list[str], criteria: str,
                        emit: Emit) -> StartupProfile:
    await emit("analyzing", {"company": startup.name, "message": f"🔬 Researching {startup.name}…"})

    async def on_search(query: str):
        await emit("search", {"agent": "deep_dive", "company": startup.name, "query": query,
                              "message": (f"📄 {startup.name}: reading their website" if query.startswith("📄")
                                          else f"🔎 {startup.name}: {query}")})

    attr_list = "\n".join(f"- {a}" for a in attributes)
    prompt = DEEP_DIVE_TASK.format(  #agent recieves these information from user and the discovery agent
        name=startup.name, website=startup.website or "unknown", one_liner=startup.one_liner,
        criteria=criteria, attributes=attr_list)
    try:
        profile = await run_agent_streaming(get_deep_dive_agent(), prompt, on_search, recursion_limit=30,
                                            schema=StartupProfile)
        profile = _normalize(profile, startup, attributes)
    except Exception as e:
        profile = _normalize(
            StartupProfile(company_name=startup.name, attributes=[],
                           summary=f"{startup.one_liner} (I couldn't verify the details for this company yet.)",
                           risks=["Research incomplete — verify the details manually."]),
            startup, attributes,
        )
        log.warning("deep dive failed for %s: %s", startup.name, e)
        await emit("warning", {"company": startup.name,
                               "message": f"⚠️ I couldn't find everything about {startup.name}"})

    await emit("company_done", {"company": startup.name, "profile": profile.model_dump(),
                                "message": f"✅ Finished {startup.name}"})
    return profile


#research them all, in parallel
async def deep_dive_all(startups: list[StartupCandidate], attributes: list[str], criteria: str,
                        emit: Emit) -> list[StartupProfile]:
    """Run deep-dive sub-agents concurrently (bounded by DEEP_DIVE_CONCURRENCY)."""
    sem = asyncio.Semaphore(get_settings().deep_dive_concurrency)

    async def guarded(s: StartupCandidate):
        async with sem:
            return await deep_dive_one(s, attributes, criteria, emit)
 
    return list(await asyncio.gather(*(guarded(s) for s in startups)))
