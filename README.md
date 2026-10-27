# 🔭 Investments Sourcing Agent (LangChain)

An AI scout for startup investments. You describe an investment thesis in a Streamlit chat-style form. A **LangChain Discovery Agent** scans the live web through **Linkup**, then **Deep-Dive sub-agents run in parallel** (one per startup) to collect the data points you asked for. The results are emailed to you as a formatted table through **SendGrid**. Progress streams to the UI in real time over **Server-Sent Events (SSE)** from a **FastAPI** backend.

```
┌──────────────┐  POST /api/enhance            ┌────────────────────────────────────────────┐
│  Streamlit   │ ────────────────────────────▶ │ FastAPI                                    │
│  frontend    │                               │                                            │
│              │  POST /api/search/stream      │  Enhancer agent (structured output)        │
│  • criteria  │ ────────────────────────────▶ │                                            │
│  • attributes│ ◀──── SSE events ──────────── │  Discovery agent ──web_search──▶ Linkup    │
│  • email     │  status / search / discovered │        │ DiscoveryResult (JSON)            │
│  • live log  │  analyzing / company_done /   │        ▼                                   │
│              │  complete / error             │  Deep-dive agents ×N (asyncio.gather)      │
└──────────────┘                               │        │ StartupProfile (JSON) each        │
                                               │        ▼                                   │
                                               │  HTML report ──▶ SendGrid ──▶ 📬           │
                                               └────────────────────────────────────────────┘
```

## Features

| Requirement | Implementation |
|---|---|
| Chatbot-based single-page UI | `frontend/app.py`: Streamlit page with chat greeting, criteria, attributes, email, and a live activity panel |
| Example prompts | Clickable example buttons that fill the criteria box |
| **Enhance with AI** | `POST /api/enhance` runs a LangChain agent with `response_format=EnhancedQuery` and writes the refined text back into the text area for review (with a *Revert* button) |
| Preset + up to 20 custom attributes | Multiselect + comma-separated input, validated on both the frontend and the backend (`SearchRequest`) |
| Linkup web search | `backend/tools.py`: async `@tool web_search` using `LinkupClient.async_search` |
| Discovery agent | `backend/agents/discovery.py`: `create_agent(..., tools=[web_search], response_format=DiscoveryResult)` |
| Parallel deep-dive agents | `backend/agents/deep_dive.py`: one agent run per startup, run with `asyncio.gather` and a semaphore (`DEEP_DIVE_CONCURRENCY`) |
| Structured outputs (JSON) | Every agent returns a Pydantic schema (`EnhancedQuery`, `DiscoveryResult`, `StartupProfile`). Results are then normalized so every startup has exactly one value per requested attribute, in order |
| Email report | `backend/email_service.py`: HTML table + analyst summaries, sent with SendGrid |
| Real-time UI (SSE) | `POST /api/search/stream` returns `text/event-stream`. The agents stream their tool calls (`agent.astream(stream_mode="values")`) so the UI shows each search query, "Found N companies", "Analyzing X", and so on |

## Project structure

```
investments-sourcing-agent/
├── backend/
│   ├── main.py            # FastAPI app + SSE endpoint
│   ├── pipeline.py        # discovery → parallel deep dives → email; EventBus for SSE
│   ├── schemas.py         # Pydantic request + structured-output schemas
│   ├── tools.py           # Linkup web_search LangChain tool
│   ├── email_service.py   # HTML report + SendGrid sender
│   ├── config.py          # env settings
│   └── agents/
│       ├── enhancer.py    # "Enhance with AI"
│       ├── discovery.py   # Discovery agent
│       ├── deep_dive.py   # Deep-dive sub-agents (parallel)
│       └── common.py      # streaming runner that reports tool calls
├── frontend/
│   └── app.py             # Streamlit UI
├── requirements.txt
├── .env.example
└── README.md
```

## Setup

**Prerequisites:** Python 3.10+, plus API keys for an LLM (free **Groq** by default; Gemini or OpenAI also work), Linkup, and SendGrid.

```bash
git clone <your-repo-url> investments-sourcing-agent
cd investments-sourcing-agent
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env    # then fill in the keys
```

### `.env` keys

| Variable | Description |
|---|---|
| `GROQ_API_KEY` | Free Groq key from https://console.groq.com/keys (default model) |
| `GOOGLE_API_KEY` / `OPENAI_API_KEY` | Only if you switch `LLM_MODEL` to Gemini or OpenAI |
| `LLM_MODEL` | Any LangChain `provider:model` string, default `groq:llama-3.3-70b-versatile`; free alternatives `groq:meta-llama/llama-4-scout-17b-16e-instruct` (higher token limit), `google_genai:gemini-3.6-flash`; paid `openai:gpt-4o-mini` |
| `LINKUP_API_KEY` | Linkup API key |
| `LINKUP_DEPTH` | `fast`, `standard` or `deep` (deep is slower and more thorough) |
| `SENDGRID_API_KEY` | SendGrid API key (Free Tier works) |
| `SENDGRID_FROM_EMAIL` | A **verified sender** in SendGrid (Settings → Sender Authentication) |
| `MAX_STARTUPS` | Default number of startups to find (the UI slider overrides it) |
| `DEEP_DIVE_CONCURRENCY` | How many deep-dive agents run at the same time (keep 1-2 on free tiers) |
| `SEARCH_MAX_RESULTS` / `SEARCH_SNIPPET_CHARS` | Size of each search result sent to the LLM; lower these if you hit token-per-minute limits |
| `BACKEND_URL` | Used by the frontend (default `http://localhost:8000`) |

## Run

Use two terminals, both from the project root with the venv active:

```bash
# 1) Backend
uvicorn backend.main:app --reload --port 8000

# 2) Frontend
streamlit run frontend/app.py
```

Open http://localhost:8501. API docs are at http://localhost:8000/docs.

### Using it
1. Type a thesis, or click an example → optionally click **✨ Enhance with AI** and review the result.
2. Pick preset attributes and add custom ones (comma-separated, max 20).
3. Enter your email → **🚀 Find startups & email report**.
4. Watch the live activity panel. When the run finishes, the results table appears and the report arrives in your inbox (check spam the first time).

## API

| Method | Path | Body | Returns |
|---|---|---|---|
| GET | `/health` | – | `{"status":"ok"}` |
| GET | `/api/attributes` | – | preset attribute list |
| POST | `/api/enhance` | `{"criteria": str}` | `{"enhanced_criteria": str, "keywords": [str]}` |
| POST | `/api/search/stream` | `{"criteria", "attributes": [str], "custom_attributes": [str], "email", "max_startups"?}` | SSE stream |

You can try the stream with curl:
```bash
curl -N -X POST localhost:8000/api/search/stream -H 'Content-Type: application/json' \
  -d '{"criteria":"Agentic AI in healthcare, Series A","attributes":["Company Name","Founders","Website"],"email":"you@x.com","max_startups":3}'
```

### SSE event types
`status` (stage change) · `search` (an agent's web query) · `discovered` (candidate list) · `analyzing` / `company_done` (per startup, with the profile) · `warning` · `error` · `complete` (all profiles + email status) · `end`.

## Design notes
- **Agents:** built with LangChain v1 `create_agent` ([docs](https://docs.langchain.com/oss/python/langchain/agents)). Each agent pairs `tools=[web_search]` with a Pydantic `response_format`, so the final answer arrives as a validated `structured_response`.
- **Live progress:** `run_agent_streaming` iterates `agent.astream(stream_mode="values")` and sends an event the first time each `web_search` tool call appears.
- **Parallelism:** the deep-dive agent is stateless, so one compiled agent serves all concurrent `ainvoke`/`astream` calls. A semaphore caps concurrency to respect rate limits.
- **Resilience:** a failed deep dive doesn't break the run. That startup is reported with "Not found" values plus a warning event. If the email fails, the results still show in the UI.
- **Consistency:** `_normalize` makes every profile have exactly the requested attributes, in the requested order, so the email table always lines up.
