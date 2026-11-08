"""Scout's tools, all in one file.

  1. WEB TOOLS (Linkup)        web_search, fetch_page      - used by the agents; cached on disk; cost counted
  2. SAFE AI CALL              structured_call             - ask the AI for an exact JSON shape, with fallbacks
  3. AI HELPERS (single calls) enhance_criteria            - the ✨ button
                               quick_answer                - short sourced answers to market questions
                               translate                   - replies in the user's language
  4. REPORT WRITER             write_overview, build_report, build_list_report

All the AI instructions live in backend/prompts.py.
"""
import json
import logging
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from langchain.tools import tool
from linkup import LinkupClient
from pydantic import BaseModel, ValidationError

from backend import prompts
from backend.agents.repair import failed_text, recover_tool_call
from backend.config import get_model, get_settings
from backend.schemas import (EnhancedQuery, QuickAnswer, RankedStartup, ReportOverview, StartupCandidate,
                             StartupProfile, Translations)

log = logging.getLogger(__name__)


# =====================================================================================================
# 1. WEB TOOLS (Linkup)  —  search ≈ $0.005, page read ≈ $0.001; repeated calls are free (disk cache)
# =====================================================================================================
PRICE_SEARCH = 0.005  # USD per standard/fast search  (deep = 0.05)
PRICE_DEEP_SEARCH = 0.05
PRICE_FETCH = 0.001   # USD per page read (no JS rendering)

_client: LinkupClient | None = None


def get_linkup_client() -> LinkupClient:
    global _client
    if _client is None:
        key = get_settings().linkup_api_key
        if not key:
            raise RuntimeError("LINKUP_API_KEY is not set")
        _client = LinkupClient(api_key=key)
    return _client


# ---------------- usage counter (shown in the backend terminal only) ----------------
usage = {"searches": 0, "page_reads": 0, "cache_hits": 0, "errors": 0}


def reset_usage():
    for k in usage:
        usage[k] = 0


def usage_cost() -> float:
    per_search = PRICE_DEEP_SEARCH if get_settings().linkup_depth == "deep" else PRICE_SEARCH
    return usage["searches"] * per_search + usage["page_reads"] * PRICE_FETCH


def usage_summary() -> str:
    def n(k, word):
        return f"{usage[k]} {word}{'' if usage[k] == 1 else ('es' if word.endswith('h') else 's')}"
    return (f"💰 Linkup usage this run: {n('searches', 'search')}, {n('page_reads', 'page read')}, "
            f"{usage['cache_hits']} answered from cache (free) ≈ ${usage_cost():.3f}")


# ---------------- disk cache ----------------
_cache: dict[str, str] | None = None
_lock = threading.Lock()


def _cache_path() -> Path:
    return Path(get_settings().cache_dir) / "linkup_cache.json"


def _load_cache() -> dict[str, str]:
    global _cache
    if _cache is None:
        try:
            _cache = json.loads(_cache_path().read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            _cache = {}
    return _cache


def _cache_get(key: str) -> str | None:
    if not get_settings().linkup_cache:
        return None
    return _load_cache().get(key)


def _cache_put(key: str, value: str):
    if not get_settings().linkup_cache:
        return
    with _lock:
        cache = _load_cache()
        cache[key] = value
        try:
            path = _cache_path()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
        except OSError as e:
            log.warning("Could not write Linkup cache: %s", e)


def _format_results(results, max_chars: int) -> str:
    lines = []
    for i, r in enumerate(results, 1):
        content = (getattr(r, "content", "") or "").replace("\n", " ")
        lines.append(f"[{i}] {r.name}\nURL: {r.url}\n{content[:max_chars]}")
    return "\n\n".join(lines) if lines else "No results."


# ---------------- the two tools the agents can call ----------------
@tool
async def web_search(query: str) -> str:
    """Search the live web with Linkup.

    Use precise queries, e.g. "Saudi Arabia web3 infrastructure startup seed round 2025"
    or "Acme Health founders Series A". Returns titles, URLs and page snippets.
    """
    s = get_settings()
    key = f"search|{s.linkup_depth}|{s.search_max_results}|{' '.join(query.lower().split())}"
    if (cached := _cache_get(key)) is not None:
        usage["cache_hits"] += 1
        return cached
    try:
        res = await get_linkup_client().async_search(
            query=query, depth=s.linkup_depth, output_type="searchResults", max_results=s.search_max_results,
        )
    except Exception as e:  # surface the error to the agent instead of crashing the run
        usage["errors"] += 1
        return f"Search failed: {str(e)[:200]}"
    usage["searches"] += 1
    text = _format_results(res.results, s.search_snippet_chars)
    _cache_put(key, text)
    return text


@tool
async def fetch_page(url: str) -> str:
    """Read the text of ONE web page with Linkup (e.g. a startup's home, about or team page).

    Use it when search snippets are not enough. Returns the page as markdown (truncated).
    """
    s = get_settings()
    key = f"fetch|{url.strip().rstrip('/').lower()}"
    if (cached := _cache_get(key)) is not None:
        usage["cache_hits"] += 1
        return cached
    try:
        # render_js=False is the cheapest option ($0.001)
        res = await get_linkup_client().async_fetch(url=url, render_js=False)
    except Exception as e:
        usage["errors"] += 1
        return f"Fetch failed: {str(e)[:200]}"
    usage["page_reads"] += 1
    text = (res.markdown or "").strip()
    limit = s.fetch_max_chars
    text = (text[:limit] + ("\n…[truncated]" if len(text) > limit else "")) if text else "Empty page."
    _cache_put(key, text)
    return text


# =====================================================================================================
# 2. SAFE AI CALL  —  one question -> one answer in an exact shape (a Pydantic schema)
# =====================================================================================================
# Small models sometimes answer in plain chat text instead of the required JSON. So we try, in order:
#   1. json_schema       the provider forces the exact shape (most reliable when supported)
#   2. function_calling  + repair of malformed tool calls
#   3. json_mode         plain JSON with the schema pasted into the prompt
#   4. text_fallback     if every attempt produced plain text, turn that text into a valid answer instead of
#                        throwing a good reply away (e.g. the Supervisor uses it as a chat reply)

def _looks_like_json(text: str) -> bool:
    t = text.strip()
    return t.startswith("{") or t.startswith("[")


async def structured_call(schema: type[BaseModel], msgs: list[tuple[str, str]],
                          text_fallback: Callable[[str], BaseModel] | None = None) -> BaseModel:
    model = get_model()
    plain_text: str | None = None
    last_exc: Exception | None = None

    def remember(exc: Exception):
        nonlocal plain_text, last_exc
        last_exc = exc
        t = failed_text(exc)
        if t and not _looks_like_json(t):
            plain_text = t

    # 1) provider-enforced JSON schema
    try:
        return await model.with_structured_output(schema, method="json_schema").ainvoke(msgs)
    except Exception as exc:  # noqa: BLE001
        remember(exc)
        log.info("json_schema failed for %s (%s)", schema.__name__, str(exc)[:160])

    # 2) tool calling (+ repair)
    try:
        return await model.with_structured_output(schema, method="function_calling").ainvoke(msgs)
    except Exception as exc:  # noqa: BLE001
        remember(exc)
        fixed = recover_tool_call(exc)
        if fixed:
            try:
                log.warning("Repaired malformed %s tool call", schema.__name__)
                return schema.model_validate(fixed["args"])
            except ValidationError:
                pass
        log.warning("function_calling failed for %s (%s)", schema.__name__, str(exc)[:160])

    # 3) JSON mode with the schema in the prompt
    hint = prompts.JSON_ONLY_HINT.format(schema=json.dumps(schema.model_json_schema()))
    msgs = [(msgs[0][0], msgs[0][1] + hint), *msgs[1:]]
    try:
        return await model.with_structured_output(schema, method="json_mode").ainvoke(msgs)
    except Exception as exc:  # noqa: BLE001
        remember(exc)
        log.warning("json_mode failed for %s (%s)", schema.__name__, str(exc)[:160])

    # 4) keep the model's plain-text answer instead of losing it
    if text_fallback and plain_text:
        log.warning("Using the model's plain-text answer for %s", schema.__name__)
        return text_fallback(plain_text)
    raise last_exc  # type: ignore[misc]


# =====================================================================================================
# 3. AI HELPERS  —  single AI calls (not agents)
# =====================================================================================================
async def enhance_criteria(raw: str) -> EnhancedQuery:
    """The ✨ button: rewrite the user's draft into a clearer, search-ready message."""
    return await structured_call(EnhancedQuery, [("system", prompts.ENHANCE_PROMPT), ("user", raw)])


ROBOT_PHRASES = [  # never let the answer sound like a machine reading a document
    (re.compile(r"\bthe (provided |available |given )?(web |search )+results?( provided)?\b", re.I), "what I found online"),
    (re.compile(r"\b(based on|according to) what I found online,?\s*", re.I), ""),
]


async def quick_answer(question: str, context: str = "") -> str:
    """A question about startups / markets / one company -> web search -> a short answer with source links.
    If the first search doesn't answer it, one better search is tried before replying."""
    convo = f"Conversation so far (only to understand who is meant):\n{context[-1500:]}\n\n" if context else ""
    queries, results = [question], [await web_search.ainvoke({"query": question})]

    async def ask() -> QuickAnswer:
        found = "\n\n".join(f"Web results for '{q}':\n{r[:4000]}" for q, r in zip(queries, results))
        return await structured_call(QuickAnswer, [
            ("system", prompts.QUICK_ANSWER_PROMPT),
            ("user", f"{convo}Question: {question}\n\n{found}")])

    ans = await ask()
    nxt = (ans.next_query or "").strip()
    if not ans.found and nxt and nxt.lower() != question.lower():
        log.info("quick answer: not found, searching again for %r", nxt)
        queries.append(nxt)
        results.append(await web_search.ainvoke({"query": nxt}))
        ans = await ask()
    text = ans.answer.strip()
    for pattern, plain in ROBOT_PHRASES:
        text = pattern.sub(plain, text)
    text = text[:1].upper() + text[1:]
    if ans.sources:
        text += "\n\n" + "  ".join(f"[source {i}]({u})" for i, u in enumerate(ans.sources[:3], 1)
                                    if str(u).startswith("http"))
    return text


async def translate(messages: list[str], lang: str) -> list[str] | None:
    """Translate chat messages into `lang` (keeps markdown, names, numbers, links). None if it fails."""
    try:
        out = await structured_call(Translations, [
            ("system", prompts.TRANSLATE_PROMPT.format(lang=lang)),
            ("user", json.dumps(messages, ensure_ascii=False))])
    except Exception as e:  # noqa: BLE001
        log.warning("translation failed: %s", e)
        return None
    return out.items if len(out.items) == len(messages) else None


# =====================================================================================================
# 4. REPORT WRITER  —  one AI call writes the judgement parts (title, summary, takeaways, ranking);
#    everything factual (tables, values, sources) is built in code from the agents' results,
#    so the report never invents data.
# =====================================================================================================
def _cell(text: str) -> str:
    return str(text or "").replace("|", "/").replace("\n", " ").strip() or "—"


def _value(p: StartupProfile, attr: str) -> str:
    for a in p.attributes:
        if a.attribute.lower() == attr.lower():
            return a.value
    return "Not found"


def _display_name(p: StartupProfile) -> str:
    v = _value(p, "Company Name")
    return v if v and v != "Not found" else p.company_name


async def write_overview(criteria: str, profiles: list[StartupProfile],
                         startups: list[StartupCandidate]) -> ReportOverview:
    why = {s.name.lower(): s.why_match for s in startups}
    compact = [{
        "name": _display_name(p),
        "why_match": why.get(p.company_name.lower(), ""),
        "summary": p.summary,
        "data": {a.attribute: a.value for a in p.attributes},
        "strengths": p.strengths, "risks": p.risks,
    } for p in profiles]
    data = json.dumps(compact, ensure_ascii=False)[:9000]
    try:
        return await structured_call(ReportOverview, [
            ("system", prompts.REPORT_PROMPT),
            ("user", f"Investment criteria: {criteria}\n\nResearch data (JSON):\n{data}"),
        ])
    except Exception as e:  # noqa: BLE001 — never lose the report because of the summary
        log.warning("Report overview failed: %s", str(e)[:200])
        return ReportOverview(
            title="Startup sourcing report",
            executive_summary=f"I researched {len(profiles)} startups matching: {criteria}.",
            key_takeaways=[], ranking=[RankedStartup(name=_display_name(p), reason="") for p in profiles])


def build_report(criteria: str, attributes: list[str], profiles: list[StartupProfile],
                 startups: list[StartupCandidate], overview: ReportOverview) -> str:
    why = {s.name.lower(): s.why_match for s in startups}
    one_liner = {s.name.lower(): s.one_liner for s in startups}
    date = datetime.now(timezone.utc).strftime("%d %B %Y")

    # order profiles by the AI ranking (unknown names keep their original order at the end)
    rank_pos = {r.name.strip().lower(): i for i, r in enumerate(overview.ranking)}
    reason = {r.name.strip().lower(): r.reason for r in overview.ranking}
    ordered = sorted(profiles, key=lambda p: min(rank_pos.get(_display_name(p).lower(), 99),
                                                 rank_pos.get(p.company_name.lower(), 99)))

    L: list[str] = [f"# {overview.title}",
                    f"*{len(profiles)} startup{'' if len(profiles) == 1 else 's'} · researched {date}*", ""]
    L += ["## Executive summary", overview.executive_summary, ""]
    if overview.key_takeaways:
        L += ["## Key takeaways", *[f"- {t}" for t in overview.key_takeaways], ""]

    # at-a-glance table (first few data points)
    cols = [a for a in attributes if a.lower() != "company name"][:4]
    L += ["## At a glance", "",
          "| # | Startup | " + " | ".join(cols) + " |",
          "|---|---|" + "---|" * len(cols)]
    for i, p in enumerate(ordered, 1):
        L.append(f"| {i} | **{_cell(_display_name(p))}** | " + " | ".join(_cell(_value(p, c)) for c in cols) + " |")
    L.append("")

    # detailed profiles
    L.append("## Company profiles")
    for i, p in enumerate(ordered, 1):
        name = _display_name(p)
        key = p.company_name.lower()
        L += ["", f"### {i}. {name}"]
        if ol := one_liner.get(key):
            L.append(f"*{ol}*")
        L.append("")
        if r := (reason.get(name.lower()) or reason.get(key)):
            L += [f"**Ranking note:** {r}", ""]
        if w := why.get(key):
            L += [f"**Why it matches:** {w}", ""]
        L += ["", f"**Overview:** {p.summary}", ""]
        if p.strengths:
            L += ["**Strengths**", *[f"- ✅ {s}" for s in p.strengths], ""]
        if p.risks:
            L += ["**Risks & open questions**", *[f"- ⚠️ {s}" for s in p.risks], ""]
        L += ["| Data point | Value | Source |", "|---|---|---|"]
        for a in p.attributes:
            src = f"[link]({a.source_url})" if a.source_url and str(a.source_url).startswith("http") else "—"
            L.append(f"| {_cell(a.attribute)} | {_cell(a.value)} | {src} |")

    L += ["", "---", "*Prepared by Scout for Arab Bank's investment team. Values marked “Not found” couldn't be "
          "verified online. Always double-check before investing.*"]
    return "\n".join(L)


def build_list_report(criteria: str, startups: list[StartupCandidate]) -> str:
    """LIST mode: a normal chat message with names + one-liners (Discovery agent only, no report)."""
    n = len(startups)
    L = [f"Here {'is the startup' if n == 1 else f'are the **{n} startups**'} I found for *{criteria}*:", ""]
    for i, s in enumerate(startups, 1):
        site = f" · [{s.website.replace('https://', '').replace('http://', '').rstrip('/')}]({s.website})" \
            if s.website and str(s.website).startswith("http") else ""
        L += [f"**{i}. {s.name}**{site}  ", f"{s.one_liner}  ", f"*Why it matches:* {s.why_match}", ""]
    L += ["Want the full picture on any of them? Say *research #1 and #3* or *research all of them*, and I'll "
          "write you a detailed report with founders, funding and more."]
    return "\n".join(L)
