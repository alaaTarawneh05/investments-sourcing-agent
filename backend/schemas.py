"""Pydantic schemas.

These are used both as FastAPI request/response models and as LangChain
`response_format` schemas, so every agent returns validated JSON.
"""
from typing import Literal, Optional

from pydantic import BaseModel, EmailStr, Field, field_validator, model_validator

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
]
MAX_CUSTOM_ATTRIBUTES = 20


# ---------- API requests ----------
class EnhanceRequest(BaseModel):
    criteria: str = Field(..., min_length=3)


class SearchRequest(BaseModel):
    criteria: str = Field(..., min_length=3)
    attributes: list[str] = Field(default_factory=list)
    custom_attributes: list[str] = Field(default_factory=list)
    email: EmailStr
    max_startups: Optional[int] = Field(default=None, ge=1, le=20)

    @field_validator("custom_attributes")
    @classmethod
    def limit_custom(cls, v: list[str]) -> list[str]:
        cleaned = [a.strip() for a in v if a and a.strip()]
        if len(cleaned) > MAX_CUSTOM_ATTRIBUTES:
            raise ValueError(f"At most {MAX_CUSTOM_ATTRIBUTES} custom attributes are allowed")
        return cleaned

    @model_validator(mode="after")
    def at_least_one_attribute(self):
        if not self.all_attributes():
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
    summary: str = Field(description="2-3 sentence investment-oriented summary")


# ---------- Chatbot ----------
DEFAULT_ATTRIBUTES = ["Company Name", "Year Founded", "Founders", "Website", "Funding Stage", "Location"]
Step = Literal["criteria", "enhance", "enhance_review", "attributes", "email", "confirm"]


class ConversationState(BaseModel):
    """Everything the chatbot has collected so far (kept by the frontend, sent each turn)."""

    criteria: Optional[str] = None
    original_criteria: Optional[str] = None
    enhanced_candidate: Optional[str] = None
    enhance_decided: bool = False
    attributes: list[str] = Field(default_factory=list)
    custom_attributes: list[str] = Field(default_factory=list)
    email: Optional[str] = None


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    message: Optional[str] = None  # free text the user typed
    action: Optional[Literal["init", "enhance", "keep", "use_enhanced", "use_original",
                             "set_attributes", "reset"]] = None  # a button the user clicked
    attributes: list[str] = Field(default_factory=list)          # payload for set_attributes
    custom_attributes: list[str] = Field(default_factory=list)
    state: ConversationState = Field(default_factory=ConversationState)
    history: list[ChatMessage] = Field(default_factory=list)


class ChatResponse(BaseModel):
    messages: list[str]         # assistant messages to show, in order
    state: ConversationState
    step: Step
    start_search: bool = False  # True when the user asked to start and everything is ready


class ChatExtraction(BaseModel):
    """Structured output of the single chat LLM call: a natural reply + any info found in the message."""

    reply: str = Field(description="Short, friendly reply (1-2 sentences) to what the user actually said. "
                                   "Do NOT ask for the next piece of information - the app adds that.")
    intent: Literal["chat", "provide_info", "start_search", "reset"] = Field(
        description="chat = greeting/question/small talk/off-topic; provide_info = user gave or changed "
                    "criteria, attributes or email; start_search = user wants to start/run/go; "
                    "reset = user wants a new search from scratch")
    criteria: Optional[str] = Field(default=None, description="The investment thesis/search criteria, ONLY if "
                                    "the user described the kind of startups they want (new or changed). "
                                    "Keep their wording.")
    use_default_attributes: bool = Field(default=False, description="User wants the default attributes")
    preset_attributes: list[str] = Field(default_factory=list, description="Preset attributes the user asked "
                                         "for, using the exact preset names")
    custom_attributes: list[str] = Field(default_factory=list, description="Data points the user asked for "
                                         "that are not in the preset list")
    remove_attributes: list[str] = Field(default_factory=list, description="Attributes the user wants removed")
    email: Optional[str] = Field(default=None, description="Email address, if the user gave one")
