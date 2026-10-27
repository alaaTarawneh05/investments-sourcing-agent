"""The chatbot's brain — ONE structured LLM call per user message (not an agent).

It does two things at once:
  1. writes a natural reply to whatever the user typed ("hi", questions, off-topic…)
  2. extracts any criteria / attributes from the message as JSON
"""
from backend.config import get_model
from backend.schemas import (DEFAULT_ATTRIBUTES, PRESET_ATTRIBUTES, ChatExtraction, ChatMessage,
                             ConversationState)

SYSTEM_PROMPT = """You are "Scout", the friendly chatbot of an Investments Sourcing app.
What the app does: the user describes the startups they want to invest in; the app searches the live web,
researches every matching startup in parallel, and shows the results right here in the chat with the data
points the user chose.

The conversation follows fixed steps: criteria -> (optional AI enhancement) -> attributes -> confirm -> search.
Current step: {step}
Collected so far:
- criteria: {criteria}
- attributes: {attributes}

Preset attributes: {presets}
Default attributes: {defaults}

How to reply:
- ALWAYS respond to what the user actually said, in 1-2 short friendly sentences, in the user's language.
  Greetings -> greet back. Questions about the app -> answer. Off-topic -> answer briefly if harmless,
  then say you're here to find startups.
- Do NOT ask for the next piece of information; the app appends the next question automatically.
- Never claim the search has started or finished.

How to extract (only what the user explicitly said in THIS message):
- criteria: only if the message describes the kind of startups/investment thesis (sector, geography, stage,
  metrics...). A greeting or a question is NOT criteria. Keep the user's wording.
- attributes: map to exact preset names when they match (e.g. "founding year" -> "Year Founded");
  anything else goes to custom_attributes. "defaults"/"the usual"/"standard ones" -> use_default_attributes.
"""


def _fmt(v):
    return v if v else "(not yet)"


async def chat_turn(message: str, step: str, state: ConversationState,
                    history: list[ChatMessage]) -> ChatExtraction:
    attrs = [*state.attributes, *state.custom_attributes]
    system = SYSTEM_PROMPT.format(
        step=step, criteria=_fmt(state.criteria), attributes=_fmt(", ".join(attrs)),
        presets=", ".join(PRESET_ATTRIBUTES), defaults=", ".join(DEFAULT_ATTRIBUTES),
    )
    msgs = [("system", system)]
    msgs += [(m.role, m.content) for m in history[-8:]]  # short memory for context
    msgs.append(("user", message))
    llm = get_model().with_structured_output(ChatExtraction, method="function_calling")
    return await llm.ainvoke(msgs)


def fallback_extraction(message: str, step: str) -> ChatExtraction:
    """Used if the LLM call fails (e.g. rate limit) so the chat never gets stuck."""
    text = message.strip()
    ext = ChatExtraction(reply="", intent="chat")
    if text.lower() in {"start", "go", "run", "yes", "start search"}:
        ext.intent = "start_search"
    elif step == "criteria" and len(text.split()) >= 3:
        ext.criteria, ext.intent = text, "provide_info"
    elif step == "attributes":
        if text.lower() in {"default", "defaults"}:
            ext.use_default_attributes = True
        else:
            ext.custom_attributes = [p.strip() for p in text.split(",") if p.strip()]
        ext.intent = "provide_info"
    return ext
