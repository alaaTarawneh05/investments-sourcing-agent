"""LangChain tools backed by the Linkup web-search API."""
from linkup import LinkupClient
from langchain.tools import tool

from backend.config import get_settings

_client: LinkupClient | None = None


def get_linkup_client() -> LinkupClient:
    global _client
    if _client is None:
        key = get_settings().linkup_api_key
        if not key:
            raise RuntimeError("LINKUP_API_KEY is not set")
        _client = LinkupClient(api_key=key)
    return _client


def _format_results(results, max_chars: int) -> str:
    lines = []
    for i, r in enumerate(results, 1):
        content = (getattr(r, "content", "") or "").replace("\n", " ")
        lines.append(f"[{i}] {r.name}\nURL: {r.url}\n{content[:max_chars]}")
    return "\n\n".join(lines) if lines else "No results."


@tool
async def web_search(query: str) -> str:
    """Search the live web with Linkup.

    Use precise queries, e.g. "Saudi Arabia web3 infrastructure startup seed round 2025"
    or "Acme Health founders Series A". Returns titles, URLs and page snippets.
    """
    client = get_linkup_client()
    try:
        res = await client.async_search(
            query=query,
            depth=get_settings().linkup_depth,
            output_type="searchResults",
            max_results=get_settings().search_max_results,
        )
    except Exception as e:  # surface the error to the agent instead of crashing the run
        return f"Search failed: {e}"
    return _format_results(res.results, get_settings().search_snippet_chars)
