"""Deep Dive Agents — one sub-agent per startup, run concurrently."""
import asyncio

from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy

from backend.agents.common import Emit, run_agent_streaming
from backend.config import get_model, get_settings
from backend.schemas import AttributeValue, StartupCandidate, StartupProfile
from backend.tools import fetch_page, web_search

SYSTEM_PROMPT = """You are a due-diligence research analyst.
You research ONE startup and collect specific data points about it.

Rules:
- Use web_search for 2-4 targeted searches (funding news, Crunchbase/LinkedIn style sources, founders).
- Use fetch_page at most 1-2 times to read the company's own website (home / about / team page) when it
  helps with founders, founding year, product or location.
- Return exactly one entry per requested attribute, using the attribute name EXACTLY as given and in
  the same order.
- Values must be concise (a few words to one sentence). Include a source_url when possible.
- If you cannot verify a value, set it to "Not found" — never guess.
- Make sure the data is about the right company (watch out for name collisions; use the website hint).
"""

_agent = None


def get_deep_dive_agent():
    global _agent
    if _agent is None:  # stateless, so a single compiled agent can serve all parallel runs
        _agent = create_agent(
            model=get_model(),
            tools=[web_search, fetch_page],
            system_prompt=SYSTEM_PROMPT,
            response_format=ToolStrategy(StartupProfile),  # works on every tool-calling model
            name="deep_dive_agent",
        )
    return _agent


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
    profile.company_name = profile.company_name or startup.name
    return profile


async def deep_dive_one(startup: StartupCandidate, attributes: list[str], criteria: str,
                        emit: Emit) -> StartupProfile:
    await emit("analyzing", {"company": startup.name, "message": f"🧠 Analyzing {startup.name}…"})

    async def on_search(query: str):
        await emit("search", {"agent": "deep_dive", "company": startup.name, "query": query,
                              "message": f"   ↳ [{startup.name}] " + (query if query.startswith("📄") else f"searching: {query}")})

    attr_list = "\n".join(f"- {a}" for a in attributes)
    prompt = (
        f"Startup: {startup.name}\n"
        f"Website hint: {startup.website or 'unknown'}\n"
        f"Description hint: {startup.one_liner}\n\n"
        f"Investment criteria (for the summary): {criteria}\n\n"
        f"Collect these attributes:\n{attr_list}"
    )
    try:
        profile = await run_agent_streaming(get_deep_dive_agent(), prompt, on_search, recursion_limit=24)
        profile = _normalize(profile, startup, attributes)
    except Exception as e:
        profile = _normalize(
            StartupProfile(company_name=startup.name, attributes=[],
                           summary=f"Deep dive failed: {e}. Discovery note: {startup.one_liner}"),
            startup, attributes,
        )
        await emit("warning", {"company": startup.name, "message": f"⚠️ {startup.name}: {e}"})

    await emit("company_done", {"company": startup.name, "profile": profile.model_dump(),
                                "message": f"✅ Finished {startup.name}"})
    return profile


async def deep_dive_all(startups: list[StartupCandidate], attributes: list[str], criteria: str,
                        emit: Emit) -> list[StartupProfile]:
    """Run deep-dive sub-agents concurrently (bounded by DEEP_DIVE_CONCURRENCY)."""
    sem = asyncio.Semaphore(get_settings().deep_dive_concurrency)

    async def guarded(s: StartupCandidate):
        async with sem:
            return await deep_dive_one(s, attributes, criteria, emit)

    return list(await asyncio.gather(*(guarded(s) for s in startups)))
