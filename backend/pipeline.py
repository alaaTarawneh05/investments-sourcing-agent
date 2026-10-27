"""End-to-end sourcing pipeline: discovery → parallel deep dives → results (streamed to the chat).

Every step pushes an event onto an asyncio.Queue, which the FastAPI SSE endpoint
streams to the Streamlit frontend in real time.
"""
import asyncio
import json
import time

from backend.agents.deep_dive import deep_dive_all
from backend.agents.discovery import discover_startups
from backend.config import get_settings
from backend.schemas import SearchRequest

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


async def run_pipeline(req: SearchRequest, bus: EventBus):
    emit = bus.emit
    try:
        attributes = req.all_attributes()
        max_startups = req.max_startups or get_settings().max_startups

        # 1. Discovery
        await emit("status", {"stage": "discovery", "message": "📡 Scanning market…"})
        discovery = await discover_startups(req.criteria, max_startups, emit)
        startups = discovery.startups
        await emit("discovered", {
            "stage": "discovery",
            "count": len(startups),
            "startups": [s.model_dump() for s in startups],
            "message": f"🏢 Found {len(startups)} companies",
        })
        if not startups:
            await emit("error", {"message": "No matching startups found. Try broadening the criteria."})
            return

        # 2. Deep dives in parallel
        await emit("status", {"stage": "deep_dive",
                              "message": f"⚡ Running {len(startups)} deep-dive agents in parallel…"})
        profiles = await deep_dive_all(startups, attributes, req.criteria, emit)

        # 3. Results go straight back to the chat
        await emit("complete", {
            "stage": "done",
            "criteria": req.criteria,
            "attributes": attributes,
            "startups": [s.model_dump() for s in startups],
            "profiles": [p.model_dump() for p in profiles],
            "message": f"🎉 Research complete — {len(profiles)} startups",
        })
    except Exception as e:
        await emit("error", {"message": f"❌ Pipeline failed: {e}"})
    finally:
        await bus.close()
