"""Chatbot conversation flow (fixed steps), driven by the chat LLM call.

Steps:  criteria -> enhance -> (enhance_review) -> attributes -> confirm -> search (results in the chat)

The LLM writes the natural reply; this module decides the step and appends the step's question,
so the conversation always moves forward in the same order.
"""
import logging

from backend.llm.chat import chat_turn, fallback_extraction
from backend.llm.enhance import enhance_criteria
from backend.schemas import (DEFAULT_ATTRIBUTES, MAX_CUSTOM_ATTRIBUTES, PRESET_ATTRIBUTES, ChatExtraction,
                             ChatRequest, ChatResponse, ConversationState)

log = logging.getLogger(__name__)

EXAMPLES = [
    "Infra space for crypto and web3, Saudi presence, Seed stage",
    "Agentic AI in healthcare, $5M+ ARR, Series A or B funding",
]

GREETING = "Hi! 👋 I'm **Scout**, your investment sourcing assistant. I search the web for startups that " \
           "match your thesis, research each one, and show you the results right here in the chat."


def compute_step(s: ConversationState) -> str:
    if not s.criteria:
        return "criteria"
    if s.enhanced_candidate and not s.enhance_decided:
        return "enhance_review"
    if not s.enhance_decided:
        return "enhance"
    if not (s.attributes or s.custom_attributes):
        return "attributes"
    return "confirm"


def summary(s: ConversationState) -> str:
    attrs = ", ".join([*s.attributes, *s.custom_attributes])
    return f"**Criteria:** {s.criteria}\n\n**Data points:** {attrs}"


def step_prompt(step: str, s: ConversationState, reminder: bool = False) -> str:
    """The question for each step. `reminder` = shorter version when the step didn't change."""
    if step == "criteria":
        if reminder:
            return "What kind of startups are you looking for? (sector, geography, stage, metrics…)"
        ex = "\n".join(f"- *{e}*" for e in EXAMPLES)
        return f"Describe the startups you're looking for — sector, geography, stage, metrics…\n\nExamples:\n{ex}"
    if step == "enhance":
        return f"Your criteria: **{s.criteria}**\n\nWant me to **✨ enhance it with AI** (add synonyms and " \
               "industry terms for a better search), or keep it as is?"
    if step == "enhance_review":
        return f"Here's the enhanced version:\n\n> {s.enhanced_candidate}\n\nUse it, or keep your original? " \
               "You can also type a new version."
    if step == "attributes":
        if reminder:
            return "Which data points should I collect? Pick below or type them (e.g. *defaults plus ARR*)."
        return ("Which **data points** should I collect for each startup?\n\n"
                f"Presets: {', '.join(PRESET_ATTRIBUTES)}\n\n"
                f"Pick them below, or just type, e.g. *\"the defaults plus ARR and number of customers\"*. "
                f"You can add up to {MAX_CUSTOM_ATTRIBUTES} custom ones.")
    if step == "confirm":
        return f"All set! Here's the plan:\n\n{summary(s)}\n\nSay **start** (or click the button) to begin, " \
               "or tell me what to change."
    return ""


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
}


def _match_preset(name: str) -> str | None:
    key = name.strip().lower()
    for p in PRESET_ATTRIBUTES:
        if p.lower() == key:
            return p
    return ALIASES.get(key)


def apply_extraction(ext: ChatExtraction, s: ConversationState, step: str) -> list[str]:
    """Update the state with what the LLM extracted. Returns notes to show the user."""
    notes: list[str] = []

    if ext.criteria and ext.criteria.strip():
        s.criteria = s.original_criteria = ext.criteria.strip()
        s.enhanced_candidate, s.enhance_decided = None, False

    # attributes
    attrs = list(s.attributes)
    custom = list(s.custom_attributes)
    if ext.use_default_attributes:
        attrs = list(dict.fromkeys([*attrs, *DEFAULT_ATTRIBUTES]))
    for a in ext.preset_attributes:
        if (p := _match_preset(a)) and p not in attrs:
            attrs.append(p)
    for a in ext.custom_attributes:
        if (p := _match_preset(a)):
            if p not in attrs:
                attrs.append(p)
        elif a.strip() and a.strip().lower() not in {c.lower() for c in custom}:
            custom.append(a.strip())
    remove = {r.strip().lower() for r in ext.remove_attributes}
    attrs = [a for a in attrs if a.lower() not in remove]
    custom = [c for c in custom if c.lower() not in remove]
    if len(custom) > MAX_CUSTOM_ATTRIBUTES:
        notes.append(f"⚠️ You can add at most {MAX_CUSTOM_ATTRIBUTES} custom attributes — I kept the first "
                     f"{MAX_CUSTOM_ATTRIBUTES}.")
        custom = custom[:MAX_CUSTOM_ATTRIBUTES]
    attributes_changed = (attrs, custom) != (s.attributes, s.custom_attributes)
    s.attributes, s.custom_attributes = attrs, custom

    # Giving later-step info while the enhance offer is open = "keep as is"
    if step in ("enhance", "enhance_review") and not ext.criteria and attributes_changed:
        s.enhance_decided, s.enhanced_candidate = True, None
    return notes


async def handle_chat(req: ChatRequest) -> ChatResponse:
    s = req.state.model_copy(deep=True)
    step_before = compute_step(s)
    out: list[str] = []
    start = False

    # ----- button actions -----
    if req.action == "init" or req.action == "reset":
        s = ConversationState()
        out = [GREETING if req.action == "init" else "Sure — let's start a new search! 🔄",
               step_prompt("criteria", s)]
        return ChatResponse(messages=out, state=s, step="criteria")

    if req.action == "enhance" and s.criteria:
        try:
            res = await enhance_criteria(s.criteria)
            s.enhanced_candidate = res.enhanced_criteria
        except Exception as e:
            log.exception("enhance failed")
            out.append(f"⚠️ I couldn't enhance it right now ({e}). Keeping your original.")
            s.enhance_decided = True
    elif req.action == "keep":
        s.enhance_decided, s.enhanced_candidate = True, None
        out.append("👍 Keeping your criteria as is.")
    elif req.action == "use_enhanced" and s.enhanced_candidate:
        s.criteria, s.enhanced_candidate, s.enhance_decided = s.enhanced_candidate, None, True
        out.append("✅ Using the enhanced criteria.")
    elif req.action == "use_original":
        s.criteria = s.original_criteria or s.criteria
        s.enhanced_candidate, s.enhance_decided = None, True
        out.append("↩️ Keeping your original criteria.")
    elif req.action == "set_attributes":
        presets = [p for a in req.attributes if (p := _match_preset(a))]
        custom = [c.strip() for c in req.custom_attributes if c.strip()]
        if len(custom) > MAX_CUSTOM_ATTRIBUTES:
            out.append(f"⚠️ Max {MAX_CUSTOM_ATTRIBUTES} custom attributes — I kept the first "
                       f"{MAX_CUSTOM_ATTRIBUTES}.")
            custom = custom[:MAX_CUSTOM_ATTRIBUTES]
        if presets or custom:
            s.attributes, s.custom_attributes = presets, custom
            out.append(f"Got it — I'll collect: {', '.join([*presets, *custom])}.")
        else:
            out.append("Please pick at least one data point.")

    # ----- free-text message -----
    elif req.message and req.message.strip():
        try:
            ext = await chat_turn(req.message, step_before, s, req.history)
        except Exception:
            log.exception("chat LLM failed; using fallback")
            ext = fallback_extraction(req.message, step_before)

        if ext.intent == "reset":
            s = ConversationState()
            return ChatResponse(messages=[ext.reply or "Starting over! 🔄", step_prompt("criteria", s)],
                                state=s, step="criteria")
        if ext.reply:
            out.append(ext.reply)
        out += apply_extraction(ext, s, step_before)
        if ext.intent == "start_search":
            if compute_step(s) == "confirm":
                start = True
            else:
                out.append("I still need a few details before I can start.")

    step = compute_step(s)
    if not start:
        out.append(step_prompt(step, s, reminder=(step == step_before and req.message is not None
                                                  and step in ("criteria", "attributes"))))
    else:
        out.append("🚀 Starting the search now…")
    return ChatResponse(messages=[m for m in out if m], state=s, step=step, start_search=start)
