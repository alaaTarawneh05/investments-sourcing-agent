"""Streamlit frontend for the Investments Sourcing Agent.

Run:  streamlit run frontend/app.py
"""
import json
import os
import re

import requests
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000").rstrip("/")
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

FALLBACK_PRESETS = ["Company Name", "Year Founded", "Founders", "Website", "Funding Stage", "Location",
                    "Total Funding Raised", "Key Investors", "Employee Count", "Product Description"]
DEFAULT_SELECTED = ["Company Name", "Year Founded", "Founders", "Website", "Funding Stage", "Location"]
EXAMPLES = [
    "Infra space for crypto and web3, Saudi presence, Seed stage",
    "Agentic AI in healthcare, $5M+ ARR, Series A or B funding",
    "Climate-tech carbon accounting SaaS in Europe, pre-seed to seed",
]

st.set_page_config(page_title="Investments Sourcing Agent", page_icon="🔭", layout="wide")


# ---------------- helpers ----------------
@st.cache_data(ttl=300)
def load_presets():
    try:
        r = requests.get(f"{BACKEND_URL}/api/attributes", timeout=5)
        r.raise_for_status()
        d = r.json()
        return d["preset"], d["max_custom"]
    except Exception:
        return FALLBACK_PRESETS, 20


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


def enhance_callback():
    raw = st.session_state.get("criteria", "").strip()
    if len(raw) < 3:
        st.session_state["enhance_msg"] = ("warning", "Write some criteria first.")
        return
    try:
        r = requests.post(f"{BACKEND_URL}/api/enhance", json={"criteria": raw}, timeout=90)
        r.raise_for_status()
        d = r.json()
        st.session_state["original_criteria"] = raw
        st.session_state["criteria"] = d["enhanced_criteria"]  # populate the text area for review
        st.session_state["enhance_msg"] = ("success", "Criteria enhanced — review & edit before running. "
                                           f"Keywords: {', '.join(d.get('keywords', []))}")
    except Exception as e:
        st.session_state["enhance_msg"] = ("error", f"Enhance failed: {e}")


def use_example(text):
    st.session_state["criteria"] = text


def revert_callback():
    if "original_criteria" in st.session_state:
        st.session_state["criteria"] = st.session_state.pop("original_criteria")
        st.session_state.pop("enhance_msg", None)


def profiles_to_rows(profiles):
    rows = []
    for p in profiles:
        row = {a["attribute"]: a["value"] for a in p["attributes"]}
        row["Summary"] = p.get("summary", "")
        rows.append(row)
    return rows


# ---------------- UI ----------------
st.session_state.setdefault("criteria", "")
presets, max_custom = load_presets()

st.title("🔭 Investments Sourcing Agent")
st.caption("Describe your thesis → the agent scans the market with live web search, "
           "deep-dives every startup in parallel, and emails you the report.")

left, right = st.columns([1.05, 1], gap="large")

with left:
    with st.chat_message("assistant", avatar="🔭"):
        st.markdown("Hi! Tell me what kind of startups you're looking for, which data points you need, "
                    "and where to send the report.")

    # A. Search criteria
    st.subheader("1 · Search criteria")
    st.markdown("**Examples** — click to use:")
    ex_cols = st.columns(len(EXAMPLES))
    for col, ex in zip(ex_cols, EXAMPLES):
        col.button(ex, on_click=use_example, args=(ex,), use_container_width=True)

    st.text_area("Investment thesis & criteria", key="criteria", height=150,
                 placeholder="e.g. Agentic AI in healthcare, $5M+ ARR, Series A or B funding")
    b1, b2 = st.columns([1, 1])
    b1.button("✨ Enhance with AI", on_click=enhance_callback, use_container_width=True)
    b2.button("↩️ Revert to original", on_click=revert_callback, use_container_width=True,
              disabled="original_criteria" not in st.session_state)
    if msg := st.session_state.get("enhance_msg"):
        getattr(st, msg[0])(msg[1])

    # B. Data collection
    st.subheader("2 · Data to collect")
    selected = st.multiselect("Preset attributes", presets,
                              default=[a for a in DEFAULT_SELECTED if a in presets])
    custom_raw = st.text_input(f"Custom attributes (comma-separated, up to {max_custom})",
                               placeholder="e.g. ARR, Customers, Tech stack, Regulatory approvals")
    custom = [c.strip() for c in custom_raw.split(",") if c.strip()]
    if len(custom) > max_custom:
        st.error(f"You entered {len(custom)} custom attributes — the maximum is {max_custom}.")

    # C. Email + run options
    st.subheader("3 · Delivery")
    email = st.text_input("Email for the report", placeholder="you@fund.com")
    max_startups = st.slider("Max startups to find", 3, 15, 5)

    can_run = (len(st.session_state["criteria"].strip()) >= 3 and (selected or custom)
               and EMAIL_RE.match(email or "") and len(custom) <= max_custom)
    run = st.button("🚀 Find startups & email report", type="primary",
                    use_container_width=True, disabled=not can_run)
    if not can_run:
        st.caption("Fill in criteria, at least one attribute and a valid email to start.")

with right:
    st.subheader("Agent activity")
    activity = st.container(height=420, border=True)
    board_ph = st.empty()
    results_ph = st.container()

    if run:
        payload = {"criteria": st.session_state["criteria"], "attributes": selected,
                   "custom_attributes": custom, "email": email, "max_startups": max_startups}
        companies: dict[str, str] = {}
        progress = st.progress(0.0, text="Starting…")
        status = st.status("Agent working…", expanded=False, state="running")

        def render_board():
            if companies:
                icons = {"queued": "⏳", "analyzing": "🧠", "done": "✅", "warning": "⚠️"}
                board_ph.markdown("**Startups**\n\n" + "\n".join(
                    f"- {icons.get(s, '•')} {n} — *{s}*" for n, s in companies.items()))

        try:
            with requests.post(f"{BACKEND_URL}/api/search/stream", json=payload,
                               stream=True, timeout=(10, 900)) as resp:
                if resp.status_code != 200:
                    st.error(f"Backend error {resp.status_code}: {resp.text}")
                    st.stop()
                for event, data in iter_sse(resp):
                    msg = data.get("message", "")
                    if event == "search":
                        activity.caption(msg)
                    elif event in ("status", "analyzing", "company_done", "discovered"):
                        activity.write(msg)
                        status.update(label=msg)
                    elif event == "warning":
                        activity.warning(msg)
                    elif event == "error":
                        activity.error(msg)
                        status.update(label=msg, state="error")

                    if event == "discovered":
                        for s in data["startups"]:
                            companies[s["name"]] = "queued"
                        progress.progress(0.15, text=msg)
                    elif event == "analyzing":
                        companies[data["company"]] = "analyzing"
                    elif event == "company_done":
                        companies[data["company"]] = "done"
                        done = sum(v == "done" for v in companies.values())
                        progress.progress(0.15 + 0.8 * done / max(len(companies), 1),
                                          text=f"Analyzed {done}/{len(companies)} startups")
                    elif event == "complete":
                        progress.progress(1.0, text=msg)
                        status.update(label=msg, state="complete" if data["email_sent"] else "error")
                        with results_ph:
                            st.subheader("Results")
                            st.dataframe(profiles_to_rows(data["profiles"]), use_container_width=True)
                            if data["email_sent"]:
                                st.success(f"Report sent to {email} 📬")
                            else:
                                st.warning(f"Email not sent: {data['email_error']}")
                            st.download_button("Download JSON", json.dumps(data["profiles"], indent=2),
                                               "startups.json", "application/json")
                    render_board()
        except requests.exceptions.ConnectionError:
            st.error(f"Cannot reach the backend at {BACKEND_URL}. Is `uvicorn backend.main:app` running?")
    else:
        activity.caption("Progress will stream here in real time once you start a search.")
