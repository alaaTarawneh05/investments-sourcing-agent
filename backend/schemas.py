"""Pydantic schemas.

These are used both as FastAPI request/response models and as LangChain
`response_format` schemas, so every agent returns validated JSON.
"""
from typing import Literal, Optional

from pydantic import BaseModel, Field, PrivateAttr, field_validator, model_validator

PRESET_ATTRIBUTES = [
    "Company Name",
    "Year Founded",
    "Founders",
    "Website",
    "Funding Stage",
    "Location",
    "Total Funding Raised",
    "Key Investors",
    "Employee Count",
    "Product Description",
    "Contact Info",
]
MAX_CUSTOM_ATTRIBUTES = 20


# ---------- API requests ----------
class SearchRequest(BaseModel):
    mode: Literal["list", "research", "lookup"] = "research"
    criteria: str = Field(..., min_length=2)
    attributes: list[str] = Field(default_factory=list)
    custom_attributes: list[str] = Field(default_factory=list)
    max_startups: Optional[int] = Field(default=None, ge=1, le=25)
    companies: list[dict] = Field(default_factory=list)  # lookup mode: [{"name", "website", "one_liner"}]

    @field_validator("custom_attributes")
    @classmethod
    def limit_custom(cls, v: list[str]) -> list[str]:
        cleaned = [a.strip() for a in v if a and a.strip()]
        if len(cleaned) > MAX_CUSTOM_ATTRIBUTES:
            raise ValueError(f"At most {MAX_CUSTOM_ATTRIBUTES} custom attributes are allowed")
        return cleaned

    @model_validator(mode="after")
    def at_least_one_attribute(self):
        if self.mode != "list" and not self.all_attributes():
            raise ValueError("Pick at least one attribute")
        return self

    def all_attributes(self) -> list[str]:
        seen, out = set(), []
        for a in [*self.attributes, *self.custom_attributes]:
            key = a.strip().lower()
            if key and key not in seen:
                seen.add(key)
                out.append(a.strip())
        return out


# ---------- Structured LLM outputs ----------
class EnhancedQuery(BaseModel):
    """Refined investment-search criteria produced by the 'Enhance with AI' step."""

    enhanced_criteria: str = Field(
        description="A rewritten, richer version of the user's investment thesis, ready to "
        "be used as search criteria. Keep every constraint the user gave (sector, geography, "
        "stage, metrics) and add synonyms / industry terms."
    )
    keywords: list[str] = Field(description="5-12 search keywords and synonyms")


class StartupCandidate(BaseModel):
    name: str = Field(description="Startup name")
    website: Optional[str] = Field(default=None, description="Official website URL if known")
    one_liner: str = Field(description="One sentence on what the company does")
    why_match: str = Field(description="Why it matches the investment criteria")


class DiscoveryResult(BaseModel):
    """Output of the discovery agent."""

    startups: list[StartupCandidate] = Field(description="Distinct startups matching the criteria")


class AttributeValue(BaseModel):
    attribute: str = Field(description="Attribute name, exactly as requested")
    value: str = Field(description="The value found, or 'Not found' if no reliable source")
    source_url: Optional[str] = Field(default=None, description="URL that supports the value")


class StartupProfile(BaseModel):
    """Output of a deep-dive agent for one startup."""

    company_name: str
    attributes: list[AttributeValue] = Field(
        description="One entry per requested attribute, in the requested order"
    )
    summary: str = Field(description="3-4 sentence investment-oriented overview of the company")
    strengths: list[str] = Field(default_factory=list,
                                 description="2-3 short investment strengths, based on the research")
    risks: list[str] = Field(default_factory=list,
                             description="1-3 short risks or open questions an investor should check")


class RankedStartup(BaseModel):
    name: str
    reason: str = Field(description="One sentence on why it ranks here")


class ReportOverview(BaseModel):
    """LLM-written parts of the final report (the data tables are built in code)."""

    title: str = Field(description="Short report title, e.g. 'Seed-stage fintech in Jordan & MENA'")
    executive_summary: str = Field(description="4-6 sentences: what was searched, what was found, overall view")
    key_takeaways: list[str] = Field(description="3-5 short bullet takeaways for an investor")
    ranking: list[RankedStartup] = Field(description="All startups ranked by fit with the criteria, best first")


# ---------- Chatbot + Supervisor ----------
DEFAULT_ATTRIBUTES = ["Company Name", "Year Founded", "Founders", "Website", "Funding Stage", "Location",
                      "Total Funding Raised", "Key Investors"]
Step = Literal["idle", "criteria", "count", "attributes", "confirm"]

# Every kind of request the Supervisor can route to
Route = Literal[
    "chat",              # greetings, thanks, small talk, questions about Scout, off-topic (linked back to startups)
    "find_list",         # "give me names of ..." -> Discovery agent ONLY (cheap)
    "research",          # full job: Discovery + Deep Dive agents + detailed report
    "company_lookup",    # user names specific companies -> Deep Dive ONLY (no discovery)
    "research_previous", # "tell me more about #2 / those two" -> Deep Dive on startups from the last results
    "follow_up",         # question answerable from previous results -> answered directly, no tools ($0)
    "quick_answer",      # general market / fact question -> 1-2 web searches + short answer
    "confirm",           # user says yes to a pending plan
    "cancel",            # user says stop / never mind
    "reset",             # user wants a brand-new search
    "decline",           # harmful / not allowed -> polite refusal
]


class MemoryStartup(BaseModel):
    """Compact memory of a startup from a previous answer (so follow-ups don't need new searches)."""

    name: str
    website: Optional[str] = None
    one_liner: str = ""
    facts: str = ""  # short "key: value; key: value" summary of researched data


class ConversationState(BaseModel):
    """What the chatbot knows (kept by the frontend, sent with every message)."""

    pending_route: Optional[Literal["find_list", "research", "company_lookup", "research_previous"]] = None
    awaiting_confirmation: bool = False
    criteria: Optional[str] = None
    num_startups: Optional[int] = None
    attributes: list[str] = Field(default_factory=list)
    custom_attributes: list[str] = Field(default_factory=list)
    companies: list[str] = Field(default_factory=list)          # for company_lookup / research_previous
    suggestion: Optional[str] = None                            # a startup idea Scout offered (e.g. weather-tech)
    focus_region: Optional[str] = None                          # e.g. "Jordan": every new search stays there
    last_batch: list[str] = Field(default_factory=list)          # names found by the LAST job only
    last_criteria: Optional[str] = None                         # memory of the last finished job
    last_startups: list[MemoryStartup] = Field(default_factory=list)


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    message: Optional[str] = None
    action: Optional[Literal["init", "reset"]] = None
    state: ConversationState = Field(default_factory=ConversationState)
    history: list[ChatMessage] = Field(default_factory=list)


class Job(BaseModel):
    """A unit of paid work the frontend should start (streams progress over SSE)."""

    mode: Literal["list", "research", "lookup"]
    criteria: str
    max_startups: Optional[int] = None
    attributes: list[str] = Field(default_factory=list)
    custom_attributes: list[str] = Field(default_factory=list)
    companies: list[MemoryStartup] = Field(default_factory=list)  # for lookup


class ChatResponse(BaseModel):
    messages: list[str]
    state: ConversationState
    step: Step
    route: str = "chat"
    job: Optional[Job] = None


class SupervisorDecision(BaseModel):
    """The Supervisor's structured decision for one user message."""

    route: Route = Field(description="Which capability should handle this message (see the routing guide)")
    on_topic: bool = Field(description="True for greetings in any language, thanks, questions about Scout itself, "
                                       "and anything about startups, companies, investors, funding or startup "
                                       "markets; false otherwise")
    continues_previous: bool = Field(default=False, description="True if this message continues the earlier "
                                     "conversation (follow-up, answer, or a change to the last request); false "
                                     "if it starts a new topic")
    reply: str = Field(description="Your chat reply. Answer the user naturally; if details are missing, ask for "
                                   "ONE of them; for follow_up put the actual answer here.")
    criteria: Optional[str] = Field(default=None, description="Kind of startups wanted (new or updated), if stated")
    num_startups: Optional[int] = Field(default=None, description="How many startups the user wants, if stated")
    companies: list[str] = Field(default_factory=list, description="Specific company names the user mentioned "
                                 "(for company_lookup / research_previous)")
    use_default_attributes: bool = Field(default=False, description="'the usual' / 'you choose' / 'anything'")
    preset_attributes: list[str] = Field(default_factory=list, description="Requested data points matching presets")
    custom_attributes: list[str] = Field(default_factory=list, description="Other requested data points")
    remove_attributes: list[str] = Field(default_factory=list, description="Data points to remove")
    question: Optional[str] = Field(default=None, description="For quick_answer: the question to research on the web")
    region: Optional[str] = Field(default=None, description="Country/region the user wants startups from, if stated "
                                  "in this message (e.g. 'Jordan'); 'anywhere' if they drop the limit")
    suggestion: Optional[str] = Field(default=None, description="When linking an off-topic message to startups: the "
                                      "startup search you offered, e.g. 'weather-tech / climate data startups'")
    _from_plain_text: bool = PrivateAttr(default=False)  # True when the model answered in plain text


class Translations(BaseModel):
    items: list[str] = Field(description="The translated messages, same order and count as the input")


class QuickAnswer(BaseModel):
    found: bool = Field(description="True if the web results contain the answer to the question")
    answer: str = Field(description="Short, factual answer (2-5 sentences) for the user, based only on the web results")
    sources: list[str] = Field(default_factory=list, description="Up to 3 source URLs used")
    next_query: Optional[str] = Field(default=None, description="If not found: ONE better web search to try next, "
                                      "e.g. '<person name> <company> LinkedIn' or '<company> contact email'")


class EnhanceRequest(BaseModel):
    criteria: str = Field(..., min_length=3)
