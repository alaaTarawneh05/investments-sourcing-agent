"""End-to-end sourcing pipeline: discovery → parallel deep dives → written report (streamed to the chat).

Every step pushes an event onto an asyncio.Queue, which the FastAPI SSE endpoint
streams to the Streamlit frontend in real time.
"""
import asyncio
import json
import logging
import time

from backend.agents.deep_dive import deep_dive_all
from backend.agents.discovery import discover_startups
from backend.config import get_settings
from backend.schemas import SearchRequest, StartupCandidate
from backend.tools import build_list_report, build_report, reset_usage, usage_summary, write_overview

log = logging.getLogger(__name__)

DONE = object()  # sentinel that closes the stream


class EventBus:
    def __init__(self):
        self.queue: asyncio.Queue = asyncio.Queue()

    async def emit(self, event: str, data: dict):
        await self.queue.put((event, {**data, "ts": time.time()}))

    async def close(self):
        await self.queue.put(DONE)


def sse_format(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"


def _memory_from_profiles(startups, profiles) -> list[dict]:
    mem = []
    for st, p in zip(startups, profiles):
        facts = "; ".join(f"{a.attribute}: {a.value}" for a in p.attributes
                          if a.value and a.value != "Not found" and a.attribute.lower() != "company name")
        mem.append({"name": st.name, "website": st.website, "one_liner": st.one_liner, "facts": facts[:350]})
    return mem


async def run_pipeline(req: SearchRequest, bus: EventBus):
    """Run one job chosen by the Supervisor:
       list     -> Discovery agent only            (names + one-liners)
       lookup   -> Deep Dive agents only           (user named the companies)
       research -> Discovery + Deep Dives + report (the full job)
    """
    emit = bus.emit
    reset_usage()
    try:
        attributes = req.all_attributes()
        s = get_settings()
        max_startups = min(req.max_startups or s.max_startups, s.max_startups_limit)

        # 1. Which startups? (discovery, or the companies the user named)
        if req.mode == "lookup":
            startups = [StartupCandidate(name=c.get("name", ""), website=c.get("website"),
                                         one_liner=c.get("one_liner") or "",
                                         why_match=(c.get("facts") or "Requested by you").replace("why it matches: ", ""))
                        for c in req.companies if c.get("name")][:s.max_startups_limit]
            await emit("discovered", {"stage": "lookup", "count": len(startups),
                                      "startups": [x.model_dump() for x in startups],
                                      "message": f"🎯 Looking into {', '.join(x.name for x in startups)}"})
        else:
            await emit("status", {"stage": "discovery", "message": "🔎 Searching the market…"})
            discovery = await discover_startups(req.criteria, max_startups, emit)
            startups = discovery.startups
            await emit("discovered", {"stage": "discovery", "count": len(startups),
                                      "startups": [x.model_dump() for x in startups],
                                      "message": f"🏢 Found {len(startups)} startup{'' if len(startups) == 1 else 's'}: "
                                                 f"{', '.join(x.name for x in startups)}"})
        if not startups:
            await emit("error", {"message": "I couldn't find matching startups. Try a broader description."})
            return

        # 2a. LIST mode stops here (no deep dives = far fewer tokens and searches)
        if req.mode == "list":
            await emit("complete", {
                "report": build_list_report(req.criteria, startups), "stage": "done", "mode": "list",
                "criteria": req.criteria,
                "memory": [{"name": x.name, "website": x.website, "one_liner": x.one_liner,
                            "facts": f"why it matches: {x.why_match}"[:350]} for x in startups],
                "message": f"🗂️ Listed {len(startups)} startups",
            })
            return

        # 2b. Deep dives in parallel
        await emit("status", {"stage": "deep_dive",
                              "message": f"🔬 Researching {'it' if len(startups) == 1 else f'all {len(startups)}'}…"})
        profiles = await deep_dive_all(startups, attributes, req.criteria, emit)

        # 3. Detailed report
        await emit("status", {"stage": "report", "message": "📝 Writing your report…"})
        overview = await write_overview(req.criteria, profiles, startups)
        report = build_report(req.criteria, attributes, profiles, startups, overview)
        await emit("complete", {
            "report": report, "stage": "done", "mode": req.mode, "criteria": req.criteria,
            "attributes": attributes, "memory": _memory_from_profiles(startups, profiles),
            "profiles": [p.model_dump() for p in profiles],
            "message": f"🎉 Research complete: {len(profiles)} startups",
        })
    except asyncio.CancelledError:
        log.info("job cancelled by the user")
        raise
    except Exception as e:
        log.exception("job failed")
        await emit("error", {"message": "Sorry, something went wrong", "detail": str(e)[:300]})
    finally:
        log.info(usage_summary())  # cost is for the developer's terminal only, never shown in the chat
        await bus.close()
