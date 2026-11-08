"""FastAPI backend.

Endpoints
- GET  /health                 liveness check
- GET  /api/attributes         preset data-point list for the UI
- POST /api/chat               chatbot turn (the Supervisor: agents/supervisor.py)
- POST /api/enhance            the ✨ button (one AI call: tools.enhance_criteria)
- POST /api/search/stream      runs the full pipeline and streams progress as Server-Sent Events
"""
import asyncio
import logging

# Developer logs (routes chosen, costs, errors) go to this terminal, never to the chat.
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s %(name)s: %(message)s",
                    datefmt="%H:%M:%S")
for noisy in ("httpx", "httpcore", "groq", "openai", "urllib3"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from backend.agents.supervisor import handle_chat
from backend.pipeline import DONE, EventBus, run_pipeline, sse_format
from backend.schemas import (DEFAULT_ATTRIBUTES, MAX_CUSTOM_ATTRIBUTES, PRESET_ATTRIBUTES, ChatRequest,
                             ChatResponse, EnhancedQuery, EnhanceRequest, SearchRequest)
from backend.tools import enhance_criteria

app = FastAPI(title="Investments Sourcing Agent", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/api/attributes")
async def attributes():
    return {"preset": PRESET_ATTRIBUTES, "default": DEFAULT_ATTRIBUTES, "max_custom": MAX_CUSTOM_ATTRIBUTES}


@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    return await handle_chat(req)


@app.post("/api/enhance", response_model=EnhancedQuery)
async def enhance(req: EnhanceRequest):
    try:
        return await enhance_criteria(req.criteria)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/search/stream")
async def search_stream(req: SearchRequest):
    bus = EventBus()
    task = asyncio.create_task(run_pipeline(req, bus))

    async def event_generator():
        yield sse_format("status", {"stage": "start", "message": "🚀 Job started"})
        try:
            while True:
                try:
                    item = await asyncio.wait_for(bus.queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    yield ": keep-alive\n\n"  # every 1s: keeps the connection open and lets the user stop quickly
                    continue
                if item is DONE:
                    break
                event, data = item
                yield sse_format(event, data)
        finally:
            if not task.done():
                task.cancel()  # client disconnected
        yield sse_format("end", {"message": "stream closed"})

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
