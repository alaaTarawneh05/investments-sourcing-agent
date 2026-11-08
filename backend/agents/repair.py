"""Repair malformed tool calls (a known quirk of gpt-oss models on Groq).

Sometimes the model names a tool call e.g. "name=DiscoveryResult" instead of "DiscoveryResult".
Groq then rejects the request with `400 tool_use_failed`, but includes the model's output in
`failed_generation`. We recover that output, fix the tool name and carry on — instead of crashing.

Used as LangChain agent middleware (`awrap_model_call`) and by the single structured LLM calls.
"""
import json
import logging
import re
import uuid
from typing import Any

from langchain.agents.middleware import AgentMiddleware, ModelResponse
from langchain.agents.structured_output import ToolStrategy
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from backend.prompts import TEXT_TO_JSON_PROMPT, TOOL_REMINDER

log = logging.getLogger(__name__)

_PREFIX_RE = re.compile(r"^(name\s*=\s*|functions\.|tools?\.)", re.I)


def _error_payload(exc: Exception) -> dict | None:
    """Return the `error` dict of a Groq/OpenAI 400 tool_use_failed error, else None."""
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        err = body.get("error", body)
        if isinstance(err, dict) and (err.get("code") == "tool_use_failed" or "failed_generation" in err):
            return err
    text = str(exc)
    if "tool_use_failed" in text or "failed_generation" in text:
        return {"code": "tool_use_failed", "message": text}
    return None


def is_tool_use_failed(exc: Exception) -> bool:
    return _error_payload(exc) is not None


def failed_text(exc: Exception) -> str | None:
    """The raw text the model produced (from `failed_generation`), if any."""
    err = _error_payload(exc)
    raw = err.get("failed_generation") if err else None
    return raw if isinstance(raw, str) and raw.strip() else None


def clean_tool_name(name: str) -> str:
    name = str(name).strip().strip("\"'` ")
    name = _PREFIX_RE.sub("", name)
    name = re.split(r"[<\s|]", name)[0]  # drop trailing junk such as "<|channel|>"
    return name.strip("\"'` ")


def recover_tool_call(exc: Exception) -> dict[str, Any] | None:
    """Parse `failed_generation` -> {"name": <fixed name>, "args": {...}} or None."""
    err = _error_payload(exc)
    if not err or not err.get("failed_generation"):
        return None
    raw = err["failed_generation"]
    try:
        data = json.loads(raw) if isinstance(raw, str) else raw
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", raw, re.S)
        if not m:
            return None
        try:
            data = json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
    if isinstance(data, list) and data:
        data = data[0]
    if not isinstance(data, dict):
        return None
    name = data.get("name") or data.get("tool") or data.get("function", {}).get("name")
    args = data.get("arguments", data.get("parameters", data.get("args")))
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            return None
    if not name or not isinstance(args, dict):
        return None
    return {"name": clean_tool_name(name), "args": args}


def _no_tool_called(exc: Exception) -> bool:
    """Groq's 'Tool choice is required, but model did not call a tool' (the AI replied with nothing usable)."""
    return "did not call a tool" in str(exc).lower()


class ToolCallRepairMiddleware(AgentMiddleware):
    """Agent middleware: when Groq rejects the AI's answer (`tool_use_failed`), fix it instead of crashing.

    1. a badly named tool call ("name=DiscoveryResult")   -> clean the name and carry on
    2. a plain-text final answer                           -> convert the text into the required JSON
    3. no tool call at all ("Tool choice is required…")    -> ask once more WITH a reminder; if it fails again,
                                                              stop: common.py then builds the answer from the
                                                              research already done (no wasted calls)
    """

    def __init__(self, agent: str = "agent", max_retries: int = 2):
        super().__init__()
        self.agent = agent  # shown in the terminal log, e.g. "Discovery agent"
        self.max_retries = max_retries

    async def awrap_model_call(self, request, handler):
        last_exc: Exception | None = None
        req, reminded = request, False
        for attempt in range(self.max_retries + 1):
            try:
                return await handler(req)
            except Exception as exc:  # noqa: BLE001
                if not is_tool_use_failed(exc):
                    raise
                last_exc = exc
                fixed = recover_tool_call(exc)
                if fixed:
                    log.warning("%s: repaired a badly formatted tool call -> %s", self.agent, fixed["name"])
                    return self._build_response(request, fixed)
                # The model answered in plain text instead of calling the output tool:
                # convert that text into the required structured output.
                converted = await self._text_to_structured(request, exc)
                if converted is not None:
                    return converted
                if _no_tool_called(exc):
                    if reminded:
                        log.warning("%s: the AI still didn't use a tool; finishing with the research done so far",
                                    self.agent)
                        break
                    final = self._final_tool_name(request) or "your final-answer tool"
                    req = request.override(messages=[*request.messages,
                                                     HumanMessage(TOOL_REMINDER.format(final=final))])
                    reminded = True
                    log.warning("%s: the AI replied without using a tool; asking again with a reminder", self.agent)
                    continue
                log.warning("%s: Groq rejected the AI's answer (attempt %d), retrying", self.agent, attempt + 1)
        raise last_exc  # type: ignore[misc]

    @staticmethod
    def _final_tool_name(request) -> str | None:
        rf = getattr(request, "response_format", None)
        return rf.schema_specs[0].name if isinstance(rf, ToolStrategy) and rf.schema_specs else None

    async def _text_to_structured(self, request, exc: Exception) -> ModelResponse | None:
        rf = getattr(request, "response_format", None)
        text = failed_text(exc)
        if not isinstance(rf, ToolStrategy) or not text or not rf.schema_specs:
            return None
        spec = rf.schema_specs[0]
        try:
            from backend.config import get_model  # local import avoids a circular import
            prompt = TEXT_TO_JSON_PROMPT.format(schema=json.dumps(spec.json_schema))
            llm = get_model().with_structured_output(spec.schema, method="json_mode")
            parsed = await llm.ainvoke([("system", prompt), ("user", text[:12000])])
        except Exception as conv_exc:  # noqa: BLE001
            log.warning("Could not convert text answer to %s: %s", spec.name, str(conv_exc)[:200])
            return None
        log.warning("Converted plain-text answer into %s", spec.name)
        return self._build_response(request, {"name": spec.name, "args": parsed.model_dump()})

    @staticmethod
    def _build_response(request, fixed: dict[str, Any]) -> ModelResponse:
        call_id = f"call_{uuid.uuid4().hex[:12]}"
        ai = AIMessage(content="", tool_calls=[{"name": fixed["name"], "args": fixed["args"],
                                                "id": call_id, "type": "tool_call"}])
        # Is it the structured-output "tool" (ToolStrategy)? Then build the structured response
        # exactly like LangChain does, so the agent finishes normally.
        rf = getattr(request, "response_format", None)
        if isinstance(rf, ToolStrategy):
            for spec in rf.schema_specs:
                if spec.name == fixed["name"]:
                    schema = spec.schema
                    parsed = schema.model_validate(fixed["args"]) if hasattr(schema, "model_validate") \
                        else fixed["args"]
                    tm = ToolMessage(content=rf.tool_message_content or
                                     f"Returning structured response: {parsed}",
                                     tool_call_id=call_id, name=fixed["name"])
                    return ModelResponse(result=[ai, tm], structured_response=parsed)
        # A normal tool (e.g. web_search): the tool node will execute it.
        return ModelResponse(result=[ai], structured_response=None)
