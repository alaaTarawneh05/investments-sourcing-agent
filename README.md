# 🔭 Scout: Investments Sourcing Agent (LangChain)

A **chatbot for Arab Bank's investment team** that scouts startups to invest in. You chat with Scout and tell it your investment thesis and the data points you want. It then:

1. runs a **Discovery agent** that searches the live web with **Linkup** to find matching startups,
2. runs one **Deep Dive agent per startup, in parallel**, to collect the data points you asked for,
3. writes a **detailed report right in the chat**, which you can save with **Download PDF**.

Every step streams live into the chat through **Server-Sent Events (SSE)** from a **FastAPI** backend, shown
as a few clear steps (Searching the market → Found 3 startups → Researching 1/3 → Writing your report).

**Stack:** Streamlit (chat UI) · FastAPI (backend) · LangChain v1 `create_agent` · Groq (free LLM) · Linkup (search)

---

## How it works

```
┌─────────────────────┐  POST /api/chat (every message)           ┌───────────────────────────────────────┐
│ Streamlit CHAT      │ ────────────────────────────────────────▶ │ FastAPI  (main.py)                    │
│ chat bubbles        │                                           │  agents/supervisor.py                 │
│ typing box ✨ ➤ ■   │ ◀────── replies + state (+ a job) ─────── │   decide (1 AI call) → follow route   │
│                     │                                           │                                       │
│                     │  POST /api/search/stream (the job)        │  pipeline.py                          │
│ live progress steps │ ────────────────────────────────────────▶ │   ├ Agent 1: Discovery  ─▶ Linkup     │
│                     │ ◀──── SSE: search / discovered / ──────── │   ├ Agent 2: Deep Dive ×N (parallel)  │
│ report + PDF        │       analyzing / company_done /          │   │      ─▶ Linkup search + fetch     │
└─────────────────────┘       complete (report + memory)          │   └ report writer (tools.py)          │
                                                                  └───────────────────────────────────────┘
```

### The conversation
Scout behaves like a normal chatbot: no forms or buttons. The typing box has **✨** (Enhance with AI:
rewrites your rough idea into a better search query you can edit), **➤** (send) and, while Scout is
working, **■** (stop).

### The Supervisor: what runs for each message
Every message first goes to the **Supervisor** (`backend/agents/supervisor.py`). One cheap structured AI
call picks a **route**, and the same file then runs *only* what that route needs, so tokens and Linkup
credit are never wasted.

| User says… | Route | What runs | Linkup cost |
|---|---|---|---|
| "hi", "thanks", "what can you do?" | `chat` | nothing, just a reply | $0 |
| "what's the weather?" (off-topic) | `chat` + suggestion | reply that links it to startups ("want weather-tech startups?") | $0 |
| "yes" to that suggestion | `confirm` | starts a research plan with that idea, asks how many | $0 |
| "give me the names of 5 fintech startups in Jordan" | `find_list` | **Discovery agent only** | ~$0.02 |
| "find 4 healthtech startups with founders and funding" | `research` | asks anything missing → confirms → **Discovery + Deep Dives + report** | ~$0.05 |
| "make it 6", "add ARR", "actually Saudi" | same route | updates the plan | $0 |
| "research #1 and #3" / "all of them" (after a list) | `research_previous` | **Deep Dive only**, reusing the list | ~$0.01 each |
| "tell me about Tabby and Tamara" | `company_lookup` | **Deep Dive only** (no discovery) | ~$0.01 each |
| "which one is in Amman?", "which raised the most?" | `follow_up` | answered from memory of the last results | $0 |
| "how big is fintech in MENA?", "what's a SAFE note?" | `quick_answer` | 1 web search + short sourced answer | ~$0.005 |
| "never mind", "stop" | `cancel` | clears the plan | $0 |
| "start over" | `reset` | fresh plan (memory kept) | $0 |
| "find the home address of …" | `decline` | polite refusal + what it can do instead | $0 |
| "try again" (after a failed job) | (frontend) | re-runs the same job; cached searches are free | less |

**House rules**
- Scout only works on startups and investing: off-topic questions ("who is the king of Jordan?", the weather,
  maths) are politely refused, never answered, and cost nothing.
- It remembers the conversation. A region you named earlier ("startups in Jordan") is kept for every later
  search until you name another one or say "any country".
- Any question about a startup is answered: from memory when it's known, otherwise with one quick web search.
  That includes the company's official contact details (email, phone, office address, LinkedIn). Private
  people's personal details are refused.
- Only Deep Dive results become a full **report** (with Download PDF); a list of names is a normal chat message.

The expensive full research **always asks for confirmation** first. Lists, lookups and quick answers start
right away. If the AI is unavailable, a rule-based backup keeps the conversation going.

### AI components
| Component | Type | LangChain | Output (structured JSON) |
|---|---|---|---|
| **Supervisor** (`backend/agents/supervisor.py`) | router: one structured AI call per message | `structured_call(SupervisorDecision)` | `route`, `reply`, criteria, count, companies, data points, question, suggestion |
| **Agent 1: Discovery** (`backend/agents/discovery.py`) | agent | `create_agent(tools=[web_search], response_format=ToolStrategy(DiscoveryResult))` | list of `{name, website, one_liner, why_match}` |
| **Agent 2: Deep Dive ×N** (`backend/agents/deep_dive.py`) | agent, one copy per startup, run with `asyncio.gather` | `create_agent(tools=[web_search, fetch_page], response_format=ToolStrategy(StartupProfile))` | `{company_name, attributes:[{attribute, value, source_url}], summary, strengths, risks}` |
| Report writer (`backend/tools.py`) | single AI call + code-built tables | `structured_call(ReportOverview)` | `title`, `executive_summary`, `key_takeaways`, `ranking` |
| Quick answer (`backend/tools.py`) | 1 web search + single AI call | `structured_call(QuickAnswer)` | `answer`, `sources` |
| Enhance (`backend/tools.py`) | single AI call (✨ in the typing box) | `structured_call(EnhancedQuery)` | `enhanced_criteria`, `keywords` |

**All the AI instructions** (every prompt) are in `backend/prompts.py`.

**Tools** (`backend/tools.py`): the LangChain `@tool`s `web_search(query)` (Linkup search: titles, URLs and
snippets) and `fetch_page(url)` (Linkup fetch: reads one page, Deep Dive only), plus the AI helpers above.

---

## Project structure
```
investments-sourcing-agent/
├── backend/
│   ├── main.py            # FastAPI: /api/chat, /api/enhance, /api/search/stream (SSE)
│   ├── prompts.py         # every AI prompt, in one place
│   ├── tools.py           # web_search + fetch_page (Linkup, cached), safe AI call, ✨ enhance,
│   │                      #   quick answers, translation, report writer
│   ├── pipeline.py        # runs a job: Discovery → parallel Deep Dives → report, emits SSE events
│   ├── schemas.py         # all Pydantic schemas (API + structured outputs)
│   ├── config.py          # .env settings + shared AI model (with retries)
│   └── agents/
│       ├── supervisor.py  # Supervisor: decides what each message needs and carries it out
│       ├── discovery.py   # Agent 1
│       ├── deep_dive.py   # Agent 2 (parallel)
│       ├── common.py      # runs an agent, streams its tool calls, salvages early stops
│       └── repair.py      # middleware that fixes the AI's badly formatted answers
├── frontend/
│   └── app.py             # Streamlit chatbot (aurora design, live steps, Download PDF)
├── tests/scenarios.py     # 43 chat scenarios against the running backend
├── .streamlit/config.toml # dark theme colours
├── requirements.txt
└── README.md
```

The **Download PDF** link is made in the browser (marked.js + html2pdf.js, loaded from the jsDelivr CDN), so
it needs no extra Python packages and handles Arabic reports correctly.

---

## Setup

### 1. Requirements
- Python **3.10+**
- 2 free API keys:

| Key | Get it from |
|---|---|
| `GROQ_API_KEY` | https://console.groq.com/keys |
| `LINKUP_API_KEY` | provided with the assignment, or https://linkup.so |


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
```

Optional settings:

| Variable | Default | Meaning |
|---|---|---|
| `LLM_MODEL` | `groq:openai/gpt-oss-120b` | Any LangChain `provider:model` string, e.g. `groq:openai/gpt-oss-20b`, `google_genai:gemini-3.6-flash`, `openai:gpt-4o-mini`. (Groq retired the Llama models for free accounts in Aug 2026.) |
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
| POST | `/api/search/stream` | `{criteria, attributes, custom_attributes, max_startups?}` → **SSE stream** (the `complete` event carries all results) |

SSE event types: `status`, `search`, `discovered`, `analyzing`, `company_done`, `warning`, `error`, `complete`, `end`.

```bash
curl -N -X POST localhost:8000/api/search/stream -H "Content-Type: application/json" \
  -d '{"criteria":"Agentic AI in healthcare, Series A","attributes":["Company Name","Founders"],"max_startups":3}'
```

---

## Design notes
- **Structured outputs everywhere.** Every LLM call and agent returns a Pydantic schema, which controls the flow (chat intent, current step) and keeps the data consistent. Deep Dive results are normalized, so each startup has exactly the requested attributes, in order, with `"Not found"` instead of guesses.
- **Parallel Deep Dives.** `asyncio.gather` with a semaphore (`DEEP_DIVE_CONCURRENCY`) to stay inside free-tier rate limits.
- **Live progress.** Agents run with `agent.astream(stream_mode="values")`. Every `web_search` / `fetch_page` call is turned into an SSE event while it happens.
- **Free-tier friendly.** The LLM retries automatically on 429 errors (`max_retries=6`). Search snippets are trimmed. If the chat LLM fails, a simple rule-based fallback keeps the conversation moving.
- **Resilient.** A failed Deep Dive doesn't crash the run. That startup is shown with "Not found" values plus a warning.

## Troubleshooting
| Problem | Fix |
|---|---|
| "I can't reach the backend" | Start terminal 1 (`uvicorn …`) |
| `ModuleNotFoundError: backend` | Run commands from the project root folder |
| Groq `429 rate limit` | Set `DEEP_DIVE_CONCURRENCY=1`, lower `MAX_STARTUPS`, or switch to `groq:openai/gpt-oss-20b` |
