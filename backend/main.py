"""FastAPI backend.

Endpoints
- GET  /health                 liveness check
- GET  /api/attributes         preset data-point list for the UI
- POST /api/enhance            'Enhance with AI' (structured output)
- POST /api/search/stream      runs the full pipeline and streams progress as Server-Sent Events
"""
import asyncio

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from backend.agents.enhancer import enhance_criteria
from backend.pipeline import DONE, EventBus, run_pipeline, sse_format
from backend.schemas import (MAX_CUSTOM_ATTRIBUTES, PRESET_ATTRIBUTES, EnhancedQuery,
                             EnhanceRequest, SearchRequest)

app = FastAPI(title="Investments Sourcing Agent", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/api/attributes")
async def attributes():
    return {"preset": PRESET_ATTRIBUTES, "max_custom": MAX_CUSTOM_ATTRIBUTES}


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
                    item = await asyncio.wait_for(bus.queue.get(), timeout=15)
                except asyncio.TimeoutError:
                    yield ": keep-alive\n\n"  # SSE comment keeps proxies from closing the connection
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
