"""'Enhance with AI' — a single structured LLM call (not an agent)."""
from backend.config import get_model
from backend.schemas import EnhancedQuery

SYSTEM_PROMPT = """You are a venture-capital research analyst.
Rewrite the user's raw investment thesis into precise, search-ready criteria.

Rules:
- Preserve EVERY constraint the user gave (sector, sub-sector, geography, stage, revenue, funding, team).
- Expand with synonyms and industry terms (e.g. "web3 infra" -> "blockchain infrastructure, node providers,
  RPC, wallets-as-a-service, L2 tooling").
- Make implicit constraints explicit (e.g. "Seed stage" -> "pre-seed or seed, raised < $5M").
- Do NOT invent constraints the user did not imply.
- Write the enhanced criteria as 2-4 clear sentences.
"""


async def enhance_criteria(raw: str) -> EnhancedQuery:
    llm = get_model().with_structured_output(EnhancedQuery, method="function_calling")
    return await llm.ainvoke([("system", SYSTEM_PROMPT), ("user", raw)])
