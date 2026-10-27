"""Pydantic schemas.

These are used both as FastAPI request/response models and as LangChain
`response_format` schemas, so every agent returns validated JSON.
"""
from typing import Optional

from pydantic import BaseModel, EmailStr, Field, field_validator

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
    attributes: list[str] = Field(..., min_length=1)
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
