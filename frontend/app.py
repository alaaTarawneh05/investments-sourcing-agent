"""Streamlit CHATBOT frontend for the Investments Sourcing Agent.

Run:  streamlit run frontend/app.py
"""
import csv
import io
import json
import os

import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000").rstrip("/")

FALLBACK_PRESETS = ["Company Name", "Year Founded", "Founders", "Website", "Funding Stage", "Location",
                    "Total Funding Raised", "Key Investors", "Employee Count", "Product Description"]
FALLBACK_DEFAULTS = FALLBACK_PRESETS[:6]
EXAMPLES = [
    "Infra space for crypto and web3, Saudi presence, Seed stage",
    "Agentic AI in healthcare, $5M+ ARR, Series A or B funding",
]

st.set_page_config(page_title="Scout · Investments Sourcing Agent", page_icon="🔭", layout="centered")


# ======================= backend helpers =======================
@st.cache_data(ttl=300)
def load_presets():
    try:
        d = requests.get(f"{BACKEND_URL}/api/attributes", timeout=5).json()
        return d["preset"], d.get("default", FALLBACK_DEFAULTS), d["max_custom"]
    except Exception:
        return FALLBACK_PRESETS, FALLBACK_DEFAULTS, 20


def history_for_llm():
    return [{"role": m["role"], "content": m["content"]}
            for m in st.session_state.messages if m.get("kind", "text") == "text"][-8:]


def call_chat(**payload):
    """Send one turn to /api/chat and append the assistant's replies."""
    body = {"state": st.session_state.conv_state, "history": history_for_llm(), **payload}
    try:
        r = requests.post(f"{BACKEND_URL}/api/chat", json=body, timeout=120)
        r.raise_for_status()
        d = r.json()
    except requests.exceptions.ConnectionError:
        add_msg("assistant", f"⚠️ I can't reach the backend at `{BACKEND_URL}`. "
                             "Start it with `uvicorn backend.main:app --port 8000`.")
        return
    except Exception as e:
        add_msg("assistant", f"⚠️ Something went wrong: {e}")
        return
    st.session_state.conv_state = d["state"]
    st.session_state.step = d["step"]
    for m in d["messages"]:
        add_msg("assistant", m)
    if d.get("start_search"):
        st.session_state.run_search = True


def iter_sse(response):
    """Minimal Server-Sent-Events parser over a streaming requests.Response."""
    event, data_lines = "message", []
    for raw in response.iter_lines(decode_unicode=True):
        if raw is None:
            continue
        line = raw.rstrip("\r")
        if line == "":
            if data_lines:
                yield event, json.loads("\n".join(data_lines))
            event, data_lines = "message", []
        elif line.startswith(":"):
            continue  # keep-alive comment
        elif line.startswith("event:"):
            event = line[6:].strip()
        elif line.startswith("data:"):
            data_lines.append(line[5:].lstrip())


# ======================= session state =======================
def add_msg(role, content, kind="text", data=None):
    st.session_state.messages.append({"role": role, "content": content, "kind": kind, "data": data})


def queue(user_text, **payload):
    """Called by buttons / chat input: show the user's bubble now, process on this rerun."""
    add_msg("user", user_text)
    st.session_state.pending = payload


ss = st.session_state
ss.setdefault("messages", [])
ss.setdefault("conv_state", {})
ss.setdefault("step", "criteria")
ss.setdefault("pending", None)
ss.setdefault("run_search", False)
presets, defaults, max_custom = load_presets()

if not ss.messages:
    call_chat(action="init")


# ======================= render conversation =======================
st.title("🔭 Scout")
st.caption("Your investment sourcing chatbot — live web search · parallel research agents · results in the chat")


def render_results(data):
    """Show the research results inside the chat: a table + one card per startup."""
    profiles = data["profiles"]
    why = {s["name"].strip().lower(): s.get("why_match", "") for s in data.get("startups", [])}

    # 1) overview table
    rows = [{a["attribute"]: a["value"] for a in p["attributes"]} for p in profiles]
    st.dataframe(rows, width="stretch", hide_index=True)

    # 2) one expandable card per startup with summary + sources
    for i, p in enumerate(profiles, 1):
        name = next((a["value"] for a in p["attributes"] if a["attribute"].lower() == "company name"
                     and a["value"] != "Not found"), p["company_name"])
        with st.expander(f"{i}. {name}"):
            st.markdown(f"**Summary:** {p.get('summary', '')}")
            if w := why.get(p["company_name"].strip().lower()) or why.get(name.strip().lower()):
                st.markdown(f"**Why it matches:** {w}")
            lines = []
            for a in p["attributes"]:
                src = f" · [source]({a['source_url']})" if a.get("source_url", "") and \
                    str(a["source_url"]).startswith("http") else ""
                lines.append(f"- **{a['attribute']}:** {a['value']}{src}")
            st.markdown("\n".join(lines))

    # 3) downloads
    c1, c2 = st.columns(2)
    c1.download_button("⬇️ Download CSV", to_csv(rows), "startups.csv", "text/csv",
                       key=f"csv_{id(data)}", width="stretch")
    c2.download_button("⬇️ Download JSON", json.dumps(profiles, indent=2), "startups.json",
                       "application/json", key=f"json_{id(data)}", width="stretch")


def to_csv(rows):
    buf = io.StringIO()
    if rows:
        w = csv.DictWriter(buf, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return buf.getvalue()


for m in ss.messages:
    with st.chat_message(m["role"], avatar="🔭" if m["role"] == "assistant" else "🧑"):
        if m["kind"] == "progress":
            with st.expander(m["content"], expanded=False):
                st.markdown("\n\n".join(m["data"]))
        elif m["kind"] == "results":
            st.markdown(m["content"])
            render_results(m["data"])
        else:
            st.markdown(m["content"])

# ======================= process a pending turn =======================
if ss.pending is not None:
    payload, ss.pending = ss.pending, None
    with st.chat_message("assistant", avatar="🔭"):
        with st.spinner("Scout is typing…"):
            call_chat(**payload)
    st.rerun()


# ======================= live search (SSE) =======================
def run_search():
    s = ss.conv_state
    payload = {"criteria": s["criteria"], "attributes": s.get("attributes", []),
               "custom_attributes": s.get("custom_attributes", [])}
    log: list[str] = []
    final = None
    with st.chat_message("assistant", avatar="🔭"):
        status = st.status("📡 Scanning market…", expanded=True)
        progress = st.progress(0.0)
        companies: dict[str, str] = {}
        try:
            with requests.post(f"{BACKEND_URL}/api/search/stream", json=payload,
                               stream=True, timeout=(10, 900)) as resp:
                if resp.status_code != 200:
                    status.update(label=f"Backend error {resp.status_code}", state="error")
                    log.append(f"❌ {resp.text}")
                for event, data in iter_sse(resp):
                    msg = data.get("message", "")
                    if event in ("search",):
                        status.caption(msg)
                        log.append(msg)
                    elif event in ("status", "discovered", "analyzing", "company_done", "warning", "error"):
                        status.write(msg)
                        status.update(label=msg)
                        log.append(msg)
                    if event == "discovered":
                        companies = {c["name"]: "queued" for c in data["startups"]}
                        if companies:
                            status.write("🏢 " + ", ".join(companies))
                        progress.progress(0.15)
                    elif event == "company_done":
                        companies[data["company"]] = "done"
                        done = sum(v == "done" for v in companies.values())
                        progress.progress(0.15 + 0.8 * done / max(len(companies), 1),
                                          text=f"Analyzed {done}/{len(companies)} startups")
                    elif event == "error":
                        status.update(label=msg, state="error")
                    elif event == "complete":
                        final = data
                        progress.progress(1.0)
                        status.update(label=msg, state="complete", expanded=False)
        except requests.exceptions.ConnectionError:
            log.append(f"❌ Cannot reach the backend at {BACKEND_URL}.")
            status.update(label="Backend not reachable", state="error")

    add_msg("assistant", f"🧾 Agent activity log ({len(log)} steps)", kind="progress", data=log or ["(empty)"])
    if final:
        n = len(final["profiles"])
        text = f"🎉 Done! I found and researched **{n} startups** matching your criteria:"
        add_msg("assistant", text, kind="results", data=final)
        add_msg("assistant", "Want to run another search? Click **New search** or just tell me what you're "
                             "looking for.")
    else:
        add_msg("assistant", "The search didn't complete. Check the activity log above, then try again.")
    ss.step = "done"


if ss.run_search:
    ss.run_search = False
    run_search()
    st.rerun()


# ======================= step-specific quick actions =======================
step = ss.step
with st.container():
    if step == "criteria":
        st.caption("Try an example:")
        for i, ex in enumerate(EXAMPLES):
            st.button(ex, key=f"ex{i}", on_click=queue, args=(ex,), kwargs={"message": ex},
                      width="stretch")

    elif step == "enhance":
        c1, c2 = st.columns(2)
        c1.button("✨ Enhance with AI", on_click=queue, args=("✨ Enhance with AI",),
                  kwargs={"action": "enhance"}, width="stretch", type="primary")
        c2.button("Keep as is", on_click=queue, args=("Keep as is",), kwargs={"action": "keep"},
                  width="stretch")

    elif step == "enhance_review":
        c1, c2 = st.columns(2)
        c1.button("✅ Use enhanced", on_click=queue, args=("Use the enhanced version",),
                  kwargs={"action": "use_enhanced"}, width="stretch", type="primary")
        c2.button("↩️ Use my original", on_click=queue, args=("Use my original",),
                  kwargs={"action": "use_original"}, width="stretch")

    elif step == "attributes":
        with st.form("attrs", border=True):
            picked = st.multiselect("Preset data points", presets, default=defaults)
            custom_raw = st.text_input(f"Custom data points (comma-separated, max {max_custom})",
                                       placeholder="ARR, Number of customers, Tech stack")
            if st.form_submit_button("Confirm data points", type="primary", width="stretch"):
                custom = [c.strip() for c in custom_raw.split(",") if c.strip()]
                if len(custom) > max_custom:
                    st.error(f"Max {max_custom} custom data points (you entered {len(custom)}).")
                elif not (picked or custom):
                    st.error("Pick at least one data point.")
                else:
                    queue("Collect: " + ", ".join([*picked, *custom]), action="set_attributes",
                          attributes=picked, custom_attributes=custom)
                    st.rerun()

    elif step == "confirm":
        c1, c2 = st.columns(2)
        if c1.button("🚀 Start search", width="stretch", type="primary"):
            add_msg("user", "🚀 Start search")
            add_msg("assistant", "🚀 Starting the search now…")
            ss.run_search = True
            st.rerun()
        c2.button("🔄 Start over", on_click=queue, args=("Start over",), kwargs={"action": "reset"},
                  width="stretch")

    elif step == "done":
        st.button("🔄 New search", on_click=queue, args=("New search",), kwargs={"action": "reset"},
                  width="stretch", type="primary")

# ======================= free-text chat input (always available) =======================
if prompt := st.chat_input("Message Scout…"):
    if ss.step == "done":  # a new message after a finished search starts a fresh conversation
        ss.conv_state, ss.step = {}, "criteria"
    queue(prompt, message=prompt)
    st.rerun()
