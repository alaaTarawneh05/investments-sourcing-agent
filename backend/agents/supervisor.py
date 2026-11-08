"""Supervisor — reads every chat message, decides what it needs, and carries that decision out.

    user message -> 1. decide()        one AI call returns a SupervisorDecision (route + reply + details)
                 -> 7. handle_chat()   follows the route: replies, asks for missing details, or returns a Job

Routes and what they run:
    chat / follow_up / decline / cancel / reset  -> nothing else, no web calls ($0)
    quick_answer                                 -> 1 web search + a short answer (~$0.005)
    find_list                                    -> Discovery agent only (~$0.02)
    company_lookup / research_previous           -> Deep Dive agents only (~$0.01 per company)
    research                                     -> Discovery + Deep Dive agents + report (~$0.05), asks "yes?" first

Why a router and not one big agent with every tool? One cheap call decides, and every paid step stays under
code control (limits, confirmation before the expensive job).

Sections:  1. decide  2. backup rules  3. the conversation  4. safety nets  5. jobs  6. language  7. main flow
The AI's instructions (SUPERVISOR_PROMPT) are in backend/prompts.py.
"""
import logging
import re

from backend.config import get_settings
from backend.prompts import SUPERVISOR_PROMPT
from backend.schemas import (DEFAULT_ATTRIBUTES, MAX_CUSTOM_ATTRIBUTES, PRESET_ATTRIBUTES, ChatMessage, ChatRequest,
                             ChatResponse, ConversationState, Job, MemoryStartup, SupervisorDecision)
from backend.tools import quick_answer, structured_call, translate

log = logging.getLogger(__name__)


# =====================================================================================================
# 1. DECIDE  —  one AI call per message: which route, what to reply, which details were given
# =====================================================================================================
def missing_for(route: str | None, s: ConversationState) -> list[str]:
    if route in ("find_list", "research"):
        miss = []
        if not s.criteria:
            miss.append("criteria")
        if not s.num_startups:
            miss.append("how many startups")
        if route == "research" and not (s.attributes or s.custom_attributes):
            miss.append("data points")
        return miss
    if route in ("company_lookup", "research_previous"):
        return [] if s.companies else ["which companies"]
    return []


def _last_results_text(s: ConversationState) -> str:
    if not s.last_startups:
        return "  (none yet)"
    lines = []
    for i, m in enumerate(s.last_startups[:12], 1):
        extra = f" | {m.facts[:220]}" if m.facts else ""
        lines.append(f"  {i}. {m.name} — {m.one_liner[:120]}{extra}")
    return "\n".join(lines)


HISTORY_TURNS = 12  # how many earlier chat messages the Supervisor reads (its memory of the conversation)


async def decide(message: str, s: ConversationState, history: list[ChatMessage], max_n: int) -> SupervisorDecision:
    attrs = [*s.attributes, *s.custom_attributes]
    system = SUPERVISOR_PROMPT.format(
        pending=s.pending_route or "none", awaiting="yes" if s.awaiting_confirmation else "no",
        criteria=s.criteria or "-", num=s.num_startups or "-", attrs=", ".join(attrs) or "-",
        companies=", ".join(s.companies) or "-",
        missing=", ".join(missing_for(s.pending_route, s)) or "nothing",
        suggestion=s.suggestion or "-", last_criteria=s.last_criteria or "none",
        last_results=_last_results_text(s), max_n=max_n, presets=", ".join(PRESET_ATTRIBUTES),
        region=s.focus_region or "none yet",
    )
    msgs = [("system", system)]
    msgs += [(m.role, m.content[:900]) for m in history[-HISTORY_TURNS:]]  # what was said earlier in the chat
    msgs.append(("user", message))
    def from_text(text: str) -> SupervisorDecision:
        d = SupervisorDecision(route="chat", reply=text.strip(), on_topic=True)
        d._from_plain_text = True
        return d
    return await structured_call(SupervisorDecision, msgs, text_fallback=from_text)


# =====================================================================================================
# 2. BACKUP RULES  —  used only if the AI call fails completely (simple word patterns, English + Arabic)
# =====================================================================================================
GREETING_RE = re.compile(r"^(hi|hello|hey|hiya|yo|salam|marhaba|good (morning|evening|afternoon)|hola|bonjour|"
                         r"salut|ciao|hallo|namaste|merhaba|ola|olá|привет|你好|こんにちは|안녕|"
                         r"مرحبا|مرحباً|أهلا|اهلا|أهلاً|هلا|هاي|السلام|صباح|مساء)", re.I)
STARTUP_HINT = re.compile(r"(startup|start-up|compan|business|fintech|tech|ai\b|saas|health|edu|ed-?tech|crypto|"
                          r"web3|invest|seed|series|fund|market|sector|industry|find|search|list|names?|research|"
                          r"شرك|ناشئ|استثمار|تقني|تكنولوجيا|ابحث|قائمة|اسماء|أسماء|قطاع|تمويل)", re.I)
QUESTION_RE = re.compile(r"^(what|how|who|why|when|where|which|can (you|u)|could (you|u)|do (you|u)|"
                         r"are (you|u)|is it|tell me)\b")
YES_RE = re.compile(r"^(yes|yeah|yep|sure|ok|okay|go|start|do it|let'?s go|go ahead)\b")
NO_RE = re.compile(r"^(no|nope|stop|cancel|never ?mind)\b")
LIST_RE = re.compile(r"\b(list|names?|just the names)\b")
WORD_NUMS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
             "nine": 9, "ten": 10, "a couple": 2, "a few": 3}


def looks_like_chat(text: str) -> bool:
    low = text.strip().lower()
    return bool(GREETING_RE.match(low) or QUESTION_RE.match(low) or low.endswith("?") or low.endswith("؟"))


def find_number(text: str) -> int | None:
    low = text.lower()
    if m := re.search(r"\b(\d{1,2})\b", low):
        return int(m.group(1))
    for w, n in WORD_NUMS.items():
        if re.search(rf"\b{w}\b", low):
            return n
    return None


def fallback_decide(message: str, s: ConversationState) -> SupervisorDecision:
    text, low = message.strip(), message.strip().lower()
    d = SupervisorDecision(route="chat", reply="", on_topic=True)
    if GREETING_RE.match(low) and not STARTUP_HINT.search(low):
        d.reply = "Hi! 👋 I'm Scout, Arab Bank's startup sourcing assistant. What kind of startups are you looking for?"
    elif YES_RE.match(low) and (s.awaiting_confirmation or s.suggestion):
        d.route = "confirm"
    elif NO_RE.match(low) and s.pending_route:
        d.route = "cancel"
    elif low in {"reset", "new search", "start over"}:
        d.route = "reset"
    elif looks_like_chat(text):
        d.reply = ("I'm Scout, Arab Bank's startup sourcing assistant. I find startups that match your investment "
                   "idea, research them on the web, and write you a detailed report here.")
    elif s.pending_route and "how many startups" in missing_for(s.pending_route, s) and find_number(text):
        d.route, d.num_startups = s.pending_route, find_number(text)
    elif s.pending_route == "research" and not (s.attributes or s.custom_attributes) and s.criteria:
        d.route = "research"
        if any(w in low for w in ("usual", "default", "you choose", "anything", "standard")):
            d.use_default_attributes = True
        else:
            d.custom_attributes = [p.strip() for p in re.split(r",| and ", text) if p.strip()]
    elif len(text.split()) >= 3 and STARTUP_HINT.search(low):
        d.route = "find_list" if LIST_RE.search(low) else "research"
        d.criteria, d.num_startups = text, find_number(text)
    return d


# =====================================================================================================
# 3. THE CONVERSATION  —  fixed texts, what to ask next, saving the details the AI extracted
# =====================================================================================================
GREETING = ("Hi! 👋 I'm **Scout**, Arab Bank's startup sourcing assistant.\n\n"
            "I can **find startups** that match your idea and write a detailed report, give you a **quick list "
            "of names**, **research specific companies**, or answer **questions about startups and their "
            "markets**.\n\n"
            "What are you looking for? For example: *agentic AI in healthcare, Series A* or "
            "*fintech in Jordan, seed stage*.")
OFF_TOPIC_REPLY = ("I'm Scout, Arab Bank's startup sourcing assistant, so I can only help with startups and "
                   "investing.")
PAID_ROUTES = ("find_list", "research", "company_lookup", "research_previous")
ANYWHERE = {"anywhere", "any", "any country", "global", "globally", "worldwide", "everywhere", "none", "no limit"}


NAMES_ONLY = re.compile(r"\b(no details?|without (the )?details|just (the )?names?|only (the )?names?|names? only|"
                        r"just names|don'?t need (the )?details)\b|بدون تفاصيل|بلا تفاصيل|بس الأسماء|بس الاسماء|"
                        r"فقط الأسماء|فقط الاسماء", re.I)
REGION_SUFFIX = re.compile(r"\s*\(only startups based in [^)]*\)\s*$", re.I)
FILLER = {"i", "want", "need", "would", "like", "please", "pls", "ok", "okay", "just", "only", "give", "me", "show",
          "the", "a", "them", "it", "now", "thanks", "and", "بدي", "بس", "لو", "سمحت", "اعطيني", "أعطيني"}
ABOUT_SCOUT = re.compile(r"\b(can|do|could) (you|u) (speak|understand|talk|chat|write)\b|\b(which|what) languages?\b|"
                         r"\bwho (are|made|built|created) (you|u)\b|\bwhat can (you|u) do\b|\bare (you|u) (a |an )?(bot|ai|human)\b|"
                         r"بتحكي|تحكي|تتكلم|مين انت|مين أنت|شو بتعمل", re.I)


def _extra_words(message: str) -> list[str]:
    """Words left after removing 'just the names' and filler: none = a pure 'names only' request."""
    rest = NAMES_ONLY.sub(" ", message.lower())
    return [w for w in re.findall(r"[^\W\d_]+", rest) if w not in FILLER]


def _suggestion_text(suggestion: str) -> str:
    """'weather-tech' -> '**weather-tech** startups'; '… tech startups' stays as it is (no 'startups startups')."""
    s = suggestion.strip().rstrip(".")
    return f"**{s}**" if "startup" in s.lower() else f"**{s}** startups"


def _names_reply(s: ConversationState) -> str:
    """'just the names' after results: list what Scout already found (free, no search)."""
    names = s.last_batch or [m.name for m in s.last_startups]
    found = [m for n in names for m in s.last_startups if m.name == n] or s.last_startups
    lines = [f"{i}. **{m.name}**" + (f": {m.one_liner}" if m.one_liner else "") for i, m in enumerate(found, 1)]
    return "Sure, just the names:\n\n" + "\n".join(lines)


def _with_region(criteria: str, region: str | None) -> str:
    """Keep every new search inside the region the user chose earlier (e.g. Jordan)."""
    if region and region.lower() not in (criteria or "").lower():
        return f"{criteria} (only startups based in {region})"
    return criteria


def compute_step(s: ConversationState) -> str:
    if not s.pending_route:
        return "idle"
    miss = missing_for(s.pending_route, s)
    if "criteria" in miss:
        return "criteria"
    if "how many startups" in miss:
        return "count"
    if "data points" in miss:
        return "attributes"
    return "confirm"


def _attrs_text(s: ConversationState) -> str:
    names = [a.lower() for a in [*s.attributes, *s.custom_attributes] if a.lower() != "company name"]
    if len(names) > 5:
        return ", ".join(names[:5]) + f" and {len(names) - 5} more"
    return ", ".join(names[:-1]) + (" and " if len(names) > 1 else "") + (names[-1] if names else "")


def question_for(s: ConversationState) -> str:
    """Plain question for the next missing detail (used when the AI's reply is missing)."""
    miss = missing_for(s.pending_route, s)
    if "criteria" in miss:
        return "What kind of startups are you looking for? Tell me the sector, region and stage."
    if "how many startups" in miss:
        return "How many startups would you like me to find?"
    if "data points" in miss:
        return ("What would you like to know about each one? For example founders, funding raised, website or "
                "year founded, or just say “the usual ones”.")
    if "which companies" in miss:
        return "Which company (or companies) should I research?"
    if s.pending_route == "research":
        where = (f" in **{s.focus_region}**" if s.focus_region and s.focus_region.lower() not in s.criteria.lower()
                 else "")
        return (f"I'll find **{s.num_startups}** startups matching *{s.criteria}*{where} and collect "
                f"{_attrs_text(s)}. Shall I start?")
    return "What would you like me to look for?"


ALIASES = {
    "name": "Company Name", "company": "Company Name", "startup name": "Company Name",
    "founding year": "Year Founded", "year": "Year Founded", "founded": "Year Founded",
    "founder": "Founders", "founders names": "Founders", "team": "Founders",
    "url": "Website", "site": "Website", "web site": "Website",
    "stage": "Funding Stage", "round": "Funding Stage", "funding round": "Funding Stage",
    "country": "Location", "hq": "Location", "headquarters": "Location", "city": "Location",
    "funding": "Total Funding Raised", "total funding": "Total Funding Raised", "amount raised": "Total Funding Raised",
    "investors": "Key Investors", "backers": "Key Investors",
    "employees": "Employee Count", "team size": "Employee Count", "headcount": "Employee Count",
    "description": "Product Description", "product": "Product Description", "what they do": "Product Description",
    "contact": "Contact Info", "contacts": "Contact Info", "contact details": "Contact Info",
    "contact information": "Contact Info", "email": "Contact Info", "phone": "Contact Info",
    "phone number": "Contact Info", "address": "Contact Info",
}


def _match_preset(name: str) -> str | None:
    key = name.strip().lower()
    for p in PRESET_ATTRIBUTES:
        if p.lower() == key:
            return p
    return ALIASES.get(key)


def apply_details(d: SupervisorDecision, s: ConversationState) -> list[str]:
    """Merge the details the Supervisor extracted into the conversation. Returns notes for the user."""
    notes: list[str] = []
    max_n = get_settings().max_startups_limit
    if d.criteria and d.criteria.strip() and not looks_like_chat(d.criteria):
        s.criteria = d.criteria.strip()
    if d.num_startups is not None:
        n = int(d.num_startups)
        if n > max_n:
            notes.append(f"I can handle up to **{max_n}** startups per request, so I'll go with {max_n}.")
            n = max_n
        if n >= 1:
            s.num_startups = n
    if d.companies:
        s.companies = list(dict.fromkeys(c.strip() for c in d.companies if c.strip()))[:max_n]

    attrs, custom = list(s.attributes), list(s.custom_attributes)
    if d.use_default_attributes:
        attrs = list(dict.fromkeys([*attrs, *DEFAULT_ATTRIBUTES]))
    for a in [*d.preset_attributes, *d.custom_attributes]:
        if (p := _match_preset(a)):
            if p not in attrs:
                attrs.append(p)
        elif a.strip() and a.strip().lower() not in {c.lower() for c in custom}:
            custom.append(a.strip())
    remove = {r.strip().lower() for r in d.remove_attributes}
    attrs = [a for a in attrs if a.lower() not in remove]
    custom = [c for c in custom if c.lower() not in remove]
    if len(custom) > MAX_CUSTOM_ATTRIBUTES:
        notes.append(f"I can collect up to {MAX_CUSTOM_ATTRIBUTES} extra data points, so I kept the first "
                     f"{MAX_CUSTOM_ATTRIBUTES}.")
        custom = custom[:MAX_CUSTOM_ATTRIBUTES]
    if (attrs or custom) and "Company Name" not in attrs:
        attrs.insert(0, "Company Name")
    s.attributes, s.custom_attributes = attrs, custom
    return notes


def _start_task(s: ConversationState, route: str):
    """Begin (or switch to) a task. A new task starts from clean details; memory is kept."""
    if s.pending_route != route:
        switching_mode = s.pending_route in ("find_list", "research") and route in ("find_list", "research")
        if not switching_mode:  # e.g. from nothing, or from a company lookup
            s.criteria, s.num_startups, s.companies = None, None, []
            s.attributes, s.custom_attributes = [], []
        s.pending_route = route
    s.awaiting_confirmation = False


def _clear_task(s: ConversationState):
    s.pending_route, s.awaiting_confirmation, s.suggestion = None, False, None
    s.criteria, s.num_startups, s.companies = None, None, []
    s.attributes, s.custom_attributes = [], []


def _match_previous(names: list[str], s: ConversationState) -> list[MemoryStartup]:
    found = []
    for n in names:
        key = n.strip().lower()
        for m in s.last_startups:
            if key and (key == m.name.lower() or key in m.name.lower() or m.name.lower() in key):
                if m not in found:
                    found.append(m)
                break
    return found


# =====================================================================================================
# 4. SAFETY NETS  —  catch the AI's mistakes ("research #2" on the wrong route, promises with no job)
# =====================================================================================================
REFERS_TO_PREVIOUS = re.compile(r"\b(them|those|these|both|the two|the three|each (one|of them)|all of them|"
                                r"the (above|list|previous|results?|ones? you found)|deep\w*|in detail|detailed|"
                                r"(full |detailed )?report|more details?|research (them|both|all))\b", re.I)
RESEARCH_WORDS = re.compile(r"\b(research|deep ?dive|dig into|more (info|details|about)|tell me more|details (on|about)|"
                            r"look into|analy[sz]e|profile|investigate)\b", re.I)
ORDINALS = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6, "seventh": 7, "eighth": 8,
            "ninth": 9, "tenth": 10, "last": -1}


def wants_research_on_previous(message: str, s: ConversationState) -> list[str]:
    """Names of previous startups the user clearly asks to research (else [])."""
    if not RESEARCH_WORDS.search(message):
        return []
    low, names = message.lower(), []
    if re.search(r"\b(all|every|each) (of )?(them|these|those|startups|companies)\b", low):
        return [m.name for m in s.last_startups]
    for n in re.findall(r"#\s*(\d{1,2})|\bnumber\s*(\d{1,2})", low):
        i = int(n[0] or n[1])
        if 1 <= i <= len(s.last_startups):
            names.append(s.last_startups[i - 1].name)
    for word, i in ORDINALS.items():
        if re.search(rf"\bthe {word}( one)?\b", low) and s.last_startups:
            names.append(s.last_startups[i - 1 if i > 0 else -1].name)
    for m in s.last_startups:
        if m.name.lower() in low:
            names.append(m.name)
    return list(dict.fromkeys(names))


DONT_KNOW_RE = re.compile(r"(don'?t have (that|this|the|those|any)[^.?!]{0,20}(detail|info|data)|"
                          r"(not|isn'?t) (in|part of) (the|my) (results|data|notes)|no (information|details) (on|about))",
                          re.I)


# A question about ONE startup's facts. If memory doesn't hold that fact, Scout searches the web instead of
# answering from memory (e.g. founders after a list of names: the list only has one-liners).
FACT_CHECKS = [
    (re.compile(r"\b(founders?|co-?founders?|ceo|cto|founded by|who (started|runs|leads|owns))\b|مؤسس", re.I),
     ("founders",)),
    (re.compile(r"linked ?in|لينكد", re.I), ("linkedin",)),
    (re.compile(r"\b(e-?mails?|phones?|contacts?|reach (them|him|her)|address|offices?)\b|"
                r"ايميل|إيميل|بريد|هاتف|رقم|تواصل|عنوان", re.I), ("contact",)),
    (re.compile(r"\b(funding|funded|raised?|investors?|backers?|valuation|rounds?)\b|تمويل|مستثمر", re.I),
     ("funding", "investors")),
    (re.compile(r"\b(employees?|team size|headcount|staff)\b|موظف", re.I), ("employee",)),
    (re.compile(r"\b(revenue|arr|profits?|customers|users)\b|ايرادات|إيرادات", re.I),
     ("revenue", "arr", "customers", "users")),
    (re.compile(r"\b(year founded|founded in|when (was|were) (it|they) founded|how old)\b", re.I), ("year founded",)),
]
POINTS_AT_ONE = re.compile(r"\b(he|him|his|she|her|it|its|this (one|company|startup)|that (one|company|startup))\b|"
                           r"(ه|ها)$", re.I)


def _target_startup(message: str, s: ConversationState, companies: list[str],
                    history: list[ChatMessage]) -> MemoryStartup | None:
    """Which ONE remembered startup a question is about: named in the message, named by the AI, or (for
    'he' / 'it' / 'their') the one mentioned last in the chat. None if it isn't about exactly one."""
    low = message.lower()
    named = [m for m in s.last_startups if m.name.lower() in low] or _match_previous(companies, s)
    if len(named) == 1:
        return named[0]
    if named:
        return None
    if len(s.last_startups) == 1:
        return s.last_startups[0]
    if POINTS_AT_ONE.search(low) or re.search(r"\b(they|them|their)\b", low):
        for h in reversed(history[-4:]):
            seen = [m for m in s.last_startups if m.name.lower() in h.content.lower()]
            if len(seen) == 1:
                return seen[0]
    return None


def _fact_not_in_memory(message: str, target: MemoryStartup | None) -> bool:
    if target is None:
        return False
    facts = (target.facts or "").lower()
    asked = [keys for rx, keys in FACT_CHECKS if rx.search(message)]
    return any(not any(k in facts for k in keys) for keys in asked)


def _name_the_company(question: str, target: MemoryStartup | None) -> str:
    """'what's their email?' -> add the startup's name so the web search finds the right company."""
    if target is None or target.name.lower() in question.lower():
        return question
    return f"{question} ({target.name}{', ' + target.website if target.website else ''})"


def _recent_chat(history: list[ChatMessage]) -> str:
    return "\n".join(f"{m.role}: {m.content[:400]}" for m in history[-4:])


PROMISE_RE = re.compile(r"(i'?m on it|on it!|get back to you|working on (it|this|that)|i'?ll (start|begin|prepare|gather|"
                        r"research|dig|look into)[^.?!]*(now|shortly|right away)|starting (the|your|now)|"
                        r"stay tuned|give me a (moment|minute|sec))", re.I)


def _asks(text: str) -> bool:
    return "?" in text or "؟" in text


def _response(out, s, route, job=None) -> ChatResponse:
    msgs = [m for m in out if m and m.strip()]
    if job is None:
        # Safety net: drop a reply that ends by announcing something that never comes ("here's the name:")
        hanging = [m for m in msgs if m.rstrip().endswith((":", "："))]
        if hanging and len(hanging) < len(msgs):
            log.warning("Removed a hanging sentence: %r", hanging[0][:80])
            msgs = [m for m in msgs if m not in hanging]
    if job is None and route != "quick_answer":
        # Safety net: Scout must never say it's working when nothing was started.
        fixed = [m for m in msgs if not PROMISE_RE.search(m)]
        if len(fixed) != len(msgs):
            log.warning("Removed a promise of work with no job (route=%s)", route)
            fixed.append(question_for(s) if s.pending_route else "What would you like me to look for?")
            msgs = list(dict.fromkeys(fixed))
    return ChatResponse(messages=msgs, state=s, step=compute_step(s), route=route, job=job)


# =====================================================================================================
# 5. JOBS  —  turn a ready plan into paid work for the agents
# =====================================================================================================
def _job_for(s: ConversationState) -> Job:
    attrs = s.attributes or list(DEFAULT_ATTRIBUTES)
    if s.pending_route == "find_list":
        return Job(mode="list", criteria=_with_region(s.criteria, s.focus_region), max_startups=s.num_startups)
    if s.pending_route == "research":
        return Job(mode="research", criteria=_with_region(s.criteria, s.focus_region), max_startups=s.num_startups,
                   attributes=attrs, custom_attributes=s.custom_attributes)
    # company_lookup / research_previous -> deep dive only
    prev = _match_previous(s.companies, s) if s.pending_route == "research_previous" else []
    known = {p.name.lower() for p in prev}
    companies = prev + [MemoryStartup(name=c) for c in s.companies
                        if c.lower() not in known and not any(c.lower() in k or k in c.lower() for k in known)]
    names = ", ".join(c.name for c in companies)
    return Job(mode="lookup", criteria=s.last_criteria if prev and s.last_criteria else f"Company research: {names}",
               max_startups=len(companies), attributes=attrs, custom_attributes=s.custom_attributes,
               companies=companies)


# =====================================================================================================
# 6. LANGUAGE  —  Arabic in, Arabic out
# =====================================================================================================
LETTERS_RE = re.compile(r"[^\W\d_]", re.UNICODE)


def detect_language(text: str) -> str | None:
    """'Arabic' / 'Persian-or-Urdu'-style scripts etc. are detected by alphabet; None = English/Latin."""
    letters = LETTERS_RE.findall(text or "")
    if not letters:
        return None
    arabic = sum(1 for c in letters if "\u0600" <= c <= "\u06FF" or "\u0750" <= c <= "\u077F")
    if arabic / len(letters) > 0.3:
        return "Arabic"
    return None


def _in_language(text: str, lang: str) -> bool:
    if lang == "Arabic":
        letters = LETTERS_RE.findall(text)
        return bool(letters) and sum(1 for c in letters if "\u0600" <= c <= "\u06FF") / len(letters) > 0.3
    return True


async def localize(resp: ChatResponse, user_message: str | None) -> ChatResponse:
    """If the user wrote in Arabic, translate any reply that isn't in Arabic (AI slip or fixed text)."""
    lang = detect_language(user_message or "")
    if not lang:
        return resp
    todo = [i for i, m in enumerate(resp.messages) if not _in_language(m, lang)]
    if not todo:
        return resp
    translated = await translate([resp.messages[i] for i in todo], lang)
    if translated:
        for i, t in zip(todo, translated):
            resp.messages[i] = t
    return resp


# =====================================================================================================
# 7. MAIN FLOW  —  what happens to one chat message, start to finish (called by main.py /api/chat)
# =====================================================================================================
async def handle_chat(req: ChatRequest) -> ChatResponse:
    return await localize(await _handle_chat(req), req.message)


async def _handle_chat(req: ChatRequest) -> ChatResponse:
    s = req.state.model_copy(deep=True)
    if req.action in ("init", "reset") or not (req.message and req.message.strip()):
        _clear_task(s)
        if req.action == "reset":
            return _response(["Sure, let's start fresh! 🔄 What are you looking for?"], s, "reset")
        return _response([GREETING], s, "chat")

    out: list[str] = []
    ai_ok = True
    try:
        d = await decide(req.message, s, req.history, get_settings().max_startups_limit)
    except Exception:  # noqa: BLE001
        log.exception("Supervisor failed; using rule-based fallback")
        ai_ok = False
        d = fallback_decide(req.message, s)
    if ai_ok and d._from_plain_text:
        rules = fallback_decide(req.message, s)
        if rules.route != "chat":  # e.g. a search request: keep the model's words, take the details from rules
            rules.reply = d.reply
            d = rules
    reply = (d.reply or "").strip()
    route = d.route
    log.info("Supervisor route=%s on_topic=%s region=%s", route, d.on_topic, d.region)

    # ---------- remember the region ("startups in Jordan" -> every later search stays in Jordan) ----------
    if d.region and d.region.strip():
        s.focus_region = None if d.region.strip().lower() in ANYWHERE else d.region.strip()

    # ---------- a greeting in any language is never off-topic ("hola", "你好") ----------
    low_msg = req.message.strip().lower()
    if GREETING_RE.match(low_msg) and len(low_msg.split()) <= 4 and not STARTUP_HINT.search(low_msg):
        d.on_topic = True
    if ABOUT_SCOUT.search(low_msg):  # "can you speak Chinese?", "who made you?" are about Scout itself
        d.on_topic = True

    # ---------- off-topic: never answer it ("who is the king of Jordan?", "what's 17 x 23?") ----------
    if not d.on_topic and route in ("chat", "quick_answer", "follow_up"):
        if d.suggestion:
            s.suggestion = d.suggestion
            return _response([f"{OFF_TOPIC_REPLY} I could look for {_suggestion_text(d.suggestion)} for you, "
                              "though. Want me to?"], s, "chat")
        out = [OFF_TOPIC_REPLY + " What kind of startups are you looking for?"]
        if s.pending_route:
            out = [OFF_TOPIC_REPLY, "Shall we get back to your search? " + question_for(s)]
        return _response(out, s, "chat")

    # ---------- safety net: "research #1" / "tell me more about <a startup we found>" ----------
    if route in ("chat", "follow_up") and s.last_startups:
        picked = wants_research_on_previous(req.message, s)
        if picked:
            log.info("Safety net: re-routing to research_previous for %s", picked)
            route, reply = "research_previous", ""
            d.companies = picked

    # ---------- a question about a startup that memory can't answer -> look it up, never guess ----------
    target = _target_startup(req.message, s, d.companies, req.history) if s.last_startups else None
    if route in ("follow_up", "chat") and s.last_startups and (
            (route == "follow_up" and DONT_KNOW_RE.search(reply)) or _fact_not_in_memory(req.message, target)):
        log.info("Safety net: the answer isn't in memory, looking it up on the web")
        route = "quick_answer"

    # ---------- no-cost routes ----------
    if route in ("chat", "decline", "follow_up"):
        if route == "chat" and d.suggestion:
            s.suggestion = d.suggestion
        if route == "follow_up" and not s.last_startups and not reply:
            reply = "I don't have any results yet. Tell me what to look for and I'll find some!"
        out.append(reply or ("I'm here to help you find and research startups. What are you looking for?"))
        if s.pending_route and not ai_ok:
            out.append(question_for(s))
        return _response(out, s, route)

    if route == "cancel":
        _clear_task(s)
        out.append(reply or "No problem, I've cancelled that. What else can I do for you?")
        return _response(out, s, route)

    if route == "reset":
        _clear_task(s)
        out.append(reply or "Sure, let's start fresh! 🔄")
        out.append(question_for(ConversationState(pending_route="research")))
        return _response(out, s, route)

    # ---------- quick web answer (~$0.005) ----------
    if route == "quick_answer":
        try:
            question = d.question if d.route == "quick_answer" and d.question else req.message
            out.append(await quick_answer(_name_the_company(question, target), context=_recent_chat(req.history)))
        except Exception as e:  # noqa: BLE001
            log.warning("quick answer failed: %s", e)
            out.append(reply or "Sorry, I couldn't look that up right now. Please try again in a moment.")
        if s.pending_route:
            out.append("Shall we get back to your search? " + question_for(s))
        return _response(out, s, route)

    # ---------- confirm: "yes" to a pending plan or to a suggestion ----------
    if route == "confirm":
        if s.pending_route and not missing_for(s.pending_route, s):
            if s.pending_route == "research" and not s.awaiting_confirmation:
                s.awaiting_confirmation = True
            job = _job_for(s)
            out.append(reply or "On it! 🔎")
            _clear_task(s)
            return _response(out, s, route, job)
        if (s.pending_route == "research" and "criteria" in missing_for("research", s) and s.last_startups):
            # e.g. "deep research" on the previous results, then "yes": research those startups
            keep_attrs = (s.attributes, s.custom_attributes)
            names = [m.name for m in s.last_startups]
            _start_task(s, "research_previous")
            s.companies, (s.attributes, s.custom_attributes) = names, keep_attrs
            job = _job_for(s)
            out.append(reply if (reply and ai_ok) else "On it! 🔬")
            _clear_task(s)
            return _response(out, s, "research_previous", job)
        if not s.pending_route and (s.suggestion or d.criteria):
            _start_task(s, "research")
            s.criteria = d.criteria or s.suggestion
            s.suggestion = None
            out += apply_details(d, s)
            out.append(f"Great, let's look at *{s.criteria}*. " + question_for(s))
            return _response(out, s, route)
        # "yes", but the plan still needs details -> ask for them (don't pretend to start)
        out.append(question_for(s) if s.pending_route else (reply or "Sure! What would you like me to look for?"))
        return _response(out, s, route)

    # ---------- "no details / just the names" right after results -> list them from memory (free) ----------
    no_new_topic = (not (d.criteria or "").strip() and not find_number(req.message)
                    and len(_extra_words(req.message)) <= 1)  # "i want no details" yes; "same but healthcare" no
    if route in ("find_list", "research") and s.last_startups and NAMES_ONLY.search(req.message) and no_new_topic:
        log.info("Names only: answering from the last results")
        return _response([_names_reply(s)], s, "follow_up")

    # ---------- a new search that continues the last one: reuse its topic instead of asking again ----------
    carried = ""
    if (route in ("find_list", "research") and not (d.criteria or "").strip() and not s.pending_route
            and d.continues_previous and s.last_criteria):
        carried = REGION_SUFFIX.sub("", s.last_criteria)
        change = " ".join(_extra_words(req.message))
        if len(change.split()) > 1:  # "same but only healthcare" -> keep the change next to the old topic
            carried = f"{carried} (change: {req.message.strip()})"
        d.criteria = carried
        log.info("Continuing the last topic: %s", carried)

    # ---------- paid routes: gather details, then start ----------
    if (route == "research" and not d.criteria and not s.criteria and s.last_startups
            and REFERS_TO_PREVIOUS.search(req.message)):
        # "deep research on them / those two / both" -> research the startups we already found
        route = "research_previous"
        n = d.num_startups or len(s.last_startups)
        d.companies = d.companies or [m.name for m in s.last_startups[:n]]
    if route in PAID_ROUTES:
        if route == "research_previous" and not s.last_startups:
            route = "company_lookup"
        _start_task(s, route)
        notes = apply_details(d, s)
        out += [n for n in notes if not (reply and str(get_settings().max_startups_limit) in reply and "up to" in n)]
        if route == "research_previous" and not _match_previous(s.companies, s) and s.companies:
            s.pending_route = "company_lookup"  # names not in the last results: research them directly
        miss = missing_for(s.pending_route, s)
        if carried:  # the AI's reply was written without the topic: use our own words
            reply = ""
            out.insert(0, f"Sticking with *{carried}*.")
        if miss:
            out.insert(0, reply) if reply and ai_ok else None
            if not reply or not ai_ok or not _asks(reply):
                out.append(question_for(s))
            return _response(out, s, route)
        if s.pending_route == "research":  # the expensive job: always confirm first
            s.awaiting_confirmation = True
            if reply and ai_ok and _asks(reply):
                out.insert(0, reply)
            else:
                out.append(question_for(s))
            return _response(out, s, route)
        job = _job_for(s)  # cheaper jobs start right away
        out.insert(0, reply if (reply and ai_ok) else "On it! 🔎")
        _clear_task(s)
        return _response(out, s, route, job)

    out.append(reply or "How can I help you with startups today?")
    return _response(out, s, route)
