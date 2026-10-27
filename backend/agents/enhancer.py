"""'Enhance with AI' — refine the user's raw investment thesis into better search criteria."""
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy

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
- Write the enhanced criteria as 2-4 clear sentences or a compact bullet list.
"""

_agent = None


def get_enhancer_agent():
    global _agent
    if _agent is None:
        _agent = create_agent(
            model=get_model(),
            tools=[],
            system_prompt=SYSTEM_PROMPT,
            response_format=ToolStrategy(EnhancedQuery),  # works on every tool-calling model
            name="criteria_enhancer",
        )
    return _agent


async def enhance_criteria(raw: str) -> EnhancedQuery:
    result = await get_enhancer_agent().ainvoke(
        {"messages": [{"role": "user", "content": raw}]}
    )
    return result["structured_response"]
