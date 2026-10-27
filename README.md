# 🔭 Scout: Investments Sourcing Agent (LangChain)

A **chatbot** that scouts startups to invest in. You chat with Scout and tell it your investment thesis, the data points you want, and your email. It then:

1. runs a **Discovery agent** that searches the live web with **Linkup** to find matching startups,
2. runs one **Deep Dive agent per startup, in parallel**, to collect the data points you asked for,
3. emails you a formatted report with **SendGrid**.

Every step streams live into the chat through **Server-Sent Events (SSE)** from a **FastAPI** backend.

**Stack:** Streamlit (chat UI) · FastAPI (backend) · LangChain v1 `create_agent` · Groq (free LLM) · Linkup (search) · SendGrid (email)

---

## How it works

```
┌─────────────────────┐  POST /api/chat (each message / button)   ┌───────────────────────────────────────┐
│ Streamlit CHAT      │ ────────────────────────────────────────▶ │ FastAPI                               │
│ st.chat_message     │                                           │  chat_flow.py  (fixed steps)          │
│ st.chat_input       │ ◀──────────── replies + state ─────────── │   └ chat LLM call  (reply + extract)  │
│ quick-reply buttons │                                           │   └ enhance LLM call                  │
│                     │  POST /api/search/stream                  │                                       │
│ live progress       │ ────────────────────────────────────────▶ │  pipeline.py                          │
│                     │ ◀──── SSE: search / discovered / ──────── │   ├ Agent 1: Discovery  ─▶ Linkup     │
│ results table       │       analyzing / company_done /          │   ├ Agent 2: Deep Dive ×N (parallel)  │
└─────────────────────┘       complete                            │   │      ─▶ Linkup search + fetch     │
                                                                  │   └ HTML report ─▶ SendGrid 📬        │
                                                                  └───────────────────────────────────────┘
```

### Conversation steps
| Step | What happens |
|---|---|
| **Greeting** | Scout introduces itself and shows example criteria (clickable) |
| **Criteria** | You describe the startups you want. Scout replies to *anything* ("hi", questions, off-topic) and steers back |
| **Enhance** | Buttons **✨ Enhance with AI** / **Keep as is**. The enhanced version is shown for review → **Use enhanced** / **Use my original** (or type a new version) |
| **Data points** | Pick presets in the picker and add up to **20 custom** ones, or just type, e.g. *"the defaults plus ARR and number of customers"* |
| **Email** | Type your email (it's validated) |
| **Confirm** | Summary → **🚀 Start search** (or type "start"), or tell Scout what to change |
| **Search** | Live progress in the chat: *Scanning market… → Searching: … → Found 5 companies → Analyzing X… → Finished X → Report emailed* |
| **Results** | Results table in the chat + **Download JSON** + **New search** |

You can also give several things at once, e.g. *"fintech in Egypt, seed, collect founders and ARR, send to me@x.com"*. Scout picks up everything and only asks for what's missing.

### AI components
| Component | Type | LangChain | Output (structured JSON) |
|---|---|---|---|
| Chat brain (`backend/llm/chat.py`) | single LLM call per message | `model.with_structured_output(ChatExtraction)` | `reply`, `intent`, `criteria`, `attributes`, `email` |
| Enhance (`backend/llm/enhance.py`) | single LLM call | `model.with_structured_output(EnhancedQuery)` | `enhanced_criteria`, `keywords` |
| **Agent 1: Discovery** (`backend/agents/discovery.py`) | agent | `create_agent(tools=[web_search], response_format=ToolStrategy(DiscoveryResult))` | list of `{name, website, one_liner, why_match}` |
| **Agent 2: Deep Dive ×N** (`backend/agents/deep_dive.py`) | agent, one copy per startup, run with `asyncio.gather` | `create_agent(tools=[web_search, fetch_page], response_format=ToolStrategy(StartupProfile))` | `{company_name, attributes:[{attribute, value, source_url}], summary}` |

**Tools** (`backend/tools.py`, LangChain `@tool`):
- `web_search(query)`: Linkup search; returns titles, URLs and snippets.
- `fetch_page(url)`: Linkup fetch; reads one page, e.g. the startup's own website (Deep Dive only).

---

## Project structure
```
investments-sourcing-agent/
├── backend/
│   ├── main.py            # FastAPI: /api/chat, /api/enhance, /api/search/stream (SSE)
│   ├── chat_flow.py       # chatbot steps + state handling
│   ├── pipeline.py        # Discovery → parallel Deep Dives → email, emits SSE events
│   ├── schemas.py         # all Pydantic schemas (API + structured outputs)
│   ├── tools.py           # Linkup web_search + fetch_page tools
│   ├── email_service.py   # HTML report + SendGrid
│   ├── config.py          # .env settings + shared LLM (with retries)
│   ├── llm/
│   │   ├── chat.py        # chat LLM call (reply + extraction) + no-LLM fallback
│   │   └── enhance.py     # "Enhance with AI" LLM call
│   └── agents/
│       ├── discovery.py   # Agent 1
│       ├── deep_dive.py   # Agent 2 (parallel)
│       └── common.py      # runs an agent and streams its tool calls
├── frontend/
│   └── app.py             # Streamlit chatbot
├── requirements.txt
├── .env.example
└── README.md
```

---

## Setup

### 1. Requirements
- Python **3.10+**
- 3 free API keys:

| Key | Get it from |
|---|---|
| `GROQ_API_KEY` | https://console.groq.com/keys |
| `LINKUP_API_KEY` | provided with the assignment, or https://linkup.so |
| `SENDGRID_API_KEY` | https://app.sendgrid.com → Settings → API Keys |

You also need a **verified sender email** in SendGrid: Settings → Sender Authentication → *Verify a Single Sender*.

### 2. Install
```bash
cd investments-sourcing-agent
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Mac / Linux:
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Configure
Copy `.env.example` to `.env` and fill in:
```
GROQ_API_KEY=gsk_...
LINKUP_API_KEY=...
SENDGRID_API_KEY=SG....
SENDGRID_FROM_EMAIL=your-verified-email@gmail.com
```

Optional settings:

| Variable | Default | Meaning |
|---|---|---|
| `LLM_MODEL` | `groq:llama-3.3-70b-versatile` | Any LangChain `provider:model` string, e.g. `groq:meta-llama/llama-4-scout-17b-16e-instruct`, `google_genai:gemini-3.6-flash`, `openai:gpt-4o-mini` |
| `MAX_STARTUPS` | 5 | Startups to find per search |
| `DEEP_DIVE_CONCURRENCY` | 2 | Deep Dive agents running at the same time |
| `LINKUP_DEPTH` | standard | `fast` / `standard` / `deep` |
| `SEARCH_MAX_RESULTS`, `SEARCH_SNIPPET_CHARS`, `FETCH_MAX_CHARS` | 5 / 500 / 3000 | How much web text goes to the LLM (lower = fewer tokens) |

## Run
Open **two terminals** in the project folder, both with the venv activated:

```bash
# Terminal 1: backend
uvicorn backend.main:app --reload --port 8000

# Terminal 2: frontend
streamlit run frontend/app.py
```
Open **http://localhost:8501** and say hi to Scout 👋. The API docs are at http://localhost:8000/docs.

---

## API
| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | health check |
| GET | `/api/attributes` | preset + default data points |
| POST | `/api/chat` | one chat turn: `{message?, action?, state, history}` → `{messages, state, step, start_search}` |
| POST | `/api/enhance` | `{criteria}` → `{enhanced_criteria, keywords}` |
| POST | `/api/search/stream` | `{criteria, attributes, custom_attributes, email, max_startups?}` → **SSE stream** |

SSE event types: `status`, `search`, `discovered`, `analyzing`, `company_done`, `warning`, `error`, `complete`, `end`.

```bash
curl -N -X POST localhost:8000/api/search/stream -H "Content-Type: application/json" \
  -d '{"criteria":"Agentic AI in healthcare, Series A","attributes":["Company Name","Founders"],"email":"you@x.com","max_startups":3}'
```

---

## Design notes
- **Structured outputs everywhere.** Every LLM call and agent returns a Pydantic schema, which controls the flow (chat intent, current step) and keeps the data consistent. Deep Dive results are normalized, so each startup has exactly the requested attributes, in order, with `"Not found"` instead of guesses.
- **Parallel Deep Dives.** `asyncio.gather` with a semaphore (`DEEP_DIVE_CONCURRENCY`) to stay inside free-tier rate limits.
- **Live progress.** Agents run with `agent.astream(stream_mode="values")`. Every `web_search` / `fetch_page` call is turned into an SSE event while it happens.
- **Free-tier friendly.** The LLM retries automatically on 429 errors (`max_retries=6`). Search snippets are trimmed. If the chat LLM fails, a simple rule-based fallback keeps the conversation moving.
- **Resilient.** A failed Deep Dive or email doesn't crash the run. You still see the results in the chat, with a warning.

## Troubleshooting
| Problem | Fix |
|---|---|
| "I can't reach the backend" | Start terminal 1 (`uvicorn …`) |
| `ModuleNotFoundError: backend` | Run commands from the project root folder |
| Groq `429 rate limit` | Set `DEEP_DIVE_CONCURRENCY=1`, lower `MAX_STARTUPS`, or use `groq:meta-llama/llama-4-scout-17b-16e-instruct` |
| SendGrid `403` | `SENDGRID_FROM_EMAIL` isn't verified |
| No email arrived | Check your spam folder |
