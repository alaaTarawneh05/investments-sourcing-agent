"""Streamlit CHATBOT frontend for the Investments Sourcing Agent ("Scout").

A normal chat — no forms, no buttons. The typing box has ✨ (Enhance with AI) and ➤ (send).
Every message goes to the backend Supervisor, which decides what to do; when it starts a job
(list / research / lookup), progress streams here live over SSE as a few clear steps, and the result
appears as a message. Every report has a "Download PDF" link under it.

Run:  streamlit run frontend/app.py      (theme: .streamlit/config.toml + the CSS below)
"""

import copy
import html
import json
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor

import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000").rstrip("/")

st.set_page_config(page_title="Scout · Startup Sourcing", page_icon="🔭", layout="centered")

ss = st.session_state
ss.setdefault("messages", [])
ss.setdefault("conv_state", {})
ss.setdefault("pending", None)
ss.setdefault("job", None)
ss.setdefault("draft", "")
ss.setdefault("composer_note", "")
ss.setdefault("turn", 0)          # increases with every user message
ss.setdefault("inflight", None)   # {"turn", "state"} while Scout is answering (so it can be stopped)
ss.setdefault("failed_job", None)
log = logging.getLogger("scout.frontend")

# ============================== DESIGN: "aurora night sky" ==============================
st.html("""
<style>
@import url("https://fonts.googleapis.com/css2?family=Sora:wght@600;700;800&family=DM+Sans:ital,opsz,wght@0,9..40,400;0,9..40,500;0,9..40,700;1,9..40,400&display=swap");
:root{
  --ink:#EAF0FF; --muted:#A9B4D6; --night:#0A0E1F; --glass:rgba(255,255,255,.055); --edge:rgba(255,255,255,.10);
  --violet:#8B5CF6; --violet-2:#A78BFA; --pink:#EC4899; --teal:#14B8A6; --teal-2:#2DD4BF; --amber:#F59E0B; --sky:#38BDF8;
  --aurora:linear-gradient(100deg,#A78BFA 0%,#F472B6 45%,#FBBF24 100%);
}
/* ---------- page + sky ---------- */
.stApp{
  background:
    radial-gradient(1100px 620px at 8% -12%, rgba(139,92,246,.42), transparent 60%),
    radial-gradient(900px 520px at 108% 6%, rgba(236,72,153,.26), transparent 62%),
    radial-gradient(1100px 760px at 50% 118%, rgba(20,184,166,.30), transparent 60%),
    var(--night) !important;
  background-attachment: fixed !important;
}
.stApp::before{ /* faint stars */
  content:""; position:fixed; inset:0; pointer-events:none; z-index:0; opacity:.55;
  background-image:
    radial-gradient(1px 1px at 20px 30px,#fff 50%,transparent 51%),
    radial-gradient(1px 1px at 120px 80px,#cfd8ff 50%,transparent 51%),
    radial-gradient(1.5px 1.5px at 200px 160px,#fff 50%,transparent 51%),
    radial-gradient(1px 1px at 280px 40px,#ffe9c7 50%,transparent 51%),
    radial-gradient(1px 1px at 340px 210px,#fff 50%,transparent 51%);
  background-size: 380px 260px;
}
header[data-testid="stHeader"]{ background:transparent !important; }
.block-container,[data-testid="stMainBlockContainer"]{ padding-top:2.2rem !important; padding-bottom:10rem !important; }
.stApp, .stApp p, .stApp li, .stApp label, .stApp input, .stApp textarea, .stApp td, .stApp th,
.stApp [data-testid="stMarkdownContainer"]{ font-family:"DM Sans", system-ui, sans-serif; }
.stApp h1,.stApp h2,.stApp h3{ font-family:"Sora","DM Sans",system-ui,sans-serif !important; letter-spacing:-.01em; }

/* ---------- hero ---------- */
.hero{ display:flex; gap:16px; align-items:center; margin-bottom:6px; }
.hero .logo{ width:58px; height:58px; border-radius:18px; display:grid; place-items:center; font-size:30px;
  background:linear-gradient(135deg,var(--violet),var(--pink)); box-shadow:0 10px 30px rgba(236,72,153,.35); flex:none; }
.hero h1{ margin:0 !important; padding:0 !important; font-size:2.3rem !important; font-weight:800 !important;
  background:var(--aurora); -webkit-background-clip:text; background-clip:text; color:transparent !important; }
.hero p{ margin:2px 0 0 !important; color:var(--muted) !important; font-size:.98rem; }
.chips{ display:flex; flex-wrap:wrap; gap:8px; margin:10px 0 18px; }
.chip{ font-size:.8rem; padding:5px 11px; border-radius:999px; border:1px solid var(--edge); background:var(--glass); color:var(--ink); }
.chip b{ font-weight:700; }
.chip.v{ border-color:rgba(167,139,250,.55);} .chip.p{ border-color:rgba(244,114,182,.55);}
.chip.t{ border-color:rgba(45,212,191,.55);} .chip.a{ border-color:rgba(251,191,36,.55);}

/* ---------- chat bubbles ---------- */
[data-testid="stElementContainer"]:has(.msg-marker){ display:none; }
[data-testid="stChatMessage"]{
  background:var(--glass); border:1px solid var(--edge); border-radius:20px; padding:14px 18px;
  backdrop-filter: blur(12px); -webkit-backdrop-filter: blur(12px); margin-bottom:12px;
  box-shadow: 0 8px 28px rgba(3,6,20,.35);
}
[data-testid="stChatMessage"]:has(.msg-user){
  flex-direction:row-reverse; width:fit-content; max-width:85%; margin-left:auto; border:none;
  background:linear-gradient(135deg, rgba(139,92,246,.95), rgba(236,72,153,.9));
  box-shadow:0 10px 30px rgba(236,72,153,.28);
}
[data-testid="stChatMessage"]:has(.msg-user) p{ color:#fff !important; }
[data-testid="stChatMessage"]:has(.msg-user) [data-testid="stChatMessageContent"]{ text-align:right; }
[data-testid="stChatMessage"]:has(.msg-report){ border:1px solid rgba(167,139,250,.35); background:rgba(17,22,48,.72); }
[data-testid="stChatMessageAvatarCustom"], [data-testid="stChatMessage"] > div:first-child:not([data-testid="stChatMessageContent"]){
  background:linear-gradient(135deg,var(--teal),var(--sky)) !important; border-radius:12px !important;
  box-shadow:0 6px 18px rgba(45,212,191,.35);
}
[data-testid="stChatMessage"]:has(.msg-user) [data-testid="stChatMessageAvatarCustom"], [data-testid="stChatMessage"] > div:first-child:not([data-testid="stChatMessageContent"]){
  background:rgba(255,255,255,.22) !important; box-shadow:none;
}
[data-testid="stChatMessage"] a{ color:#7DD3FC !important; }
[data-testid="stChatMessage"] h1{ font-size:1.6rem !important; background:var(--aurora); -webkit-background-clip:text;
  background-clip:text; color:transparent !important; padding-top:.2rem !important; }
[data-testid="stChatMessage"] h2{ font-size:1.15rem !important; color:var(--teal-2) !important; margin-top:1.1rem !important; }
[data-testid="stChatMessage"] h3{ font-size:1.02rem !important; color:var(--violet-2) !important; }
[data-testid="stChatMessage"] table{ border-collapse:separate !important; border-spacing:0; width:100%;
  border:1px solid var(--edge); border-radius:12px; overflow:hidden; font-size:.88rem; }
[data-testid="stChatMessage"] th{ background:rgba(139,92,246,.28) !important; color:#fff !important; }
[data-testid="stChatMessage"] td, [data-testid="stChatMessage"] th{ border:none !important; border-bottom:1px solid var(--edge) !important; padding:7px 10px !important; }
[data-testid="stChatMessage"] tr:nth-child(even) td{ background:rgba(255,255,255,.035); }
[data-testid="stChatMessage"] hr{ border-color:var(--edge) !important; }

/* ---------- live progress ---------- */
[data-testid="stExpander"] details{ background:rgba(10,14,31,.55) !important; border:1px solid rgba(45,212,191,.4) !important; border-radius:14px !important; }
[data-testid="stExpander"] summary{ font-weight:600; }
.stProgress > div > div > div > div{ background:linear-gradient(90deg,var(--violet),var(--pink),var(--amber)) !important; }
[data-testid="stSpinner"] i{ border-top-color:var(--pink) !important; }

/* ---------- the typing box, pinned to the bottom ---------- */
.st-key-composer{
  position:fixed; left:0; right:0; bottom:0; z-index:99;
  padding:18px max(16px, calc((100vw - 736px) / 2)) calc(18px + env(safe-area-inset-bottom, 0px));
  background:linear-gradient(180deg, rgba(10,14,31,0) 0%, rgba(10,14,31,.92) 38%, #0A0E1F 100%);
}
.st-key-composer [data-testid="stForm"]{
  border:1.5px solid transparent !important; border-radius:28px; padding:5px 7px 5px 18px;
  background:linear-gradient(#121733,#121733) padding-box, linear-gradient(100deg,#8B5CF6,#EC4899,#14B8A6) border-box;
  box-shadow:0 12px 40px rgba(139,92,246,.30);
}
.st-key-composer [data-baseweb="input"], .st-key-composer [data-baseweb="base-input"]{ border:none !important; background:transparent !important; }
.st-key-composer [data-testid="stTextInput"] div, .st-key-composer [data-testid="stTextInputRootElement"]{
  background:transparent !important; border-color:transparent !important; box-shadow:none !important; }
.st-key-composer input{ background:transparent !important; font-size:1.02rem !important; color:var(--ink) !important; }
.st-key-composer input::placeholder{ color:#8C97BD !important; }
.st-key-composer [data-testid="stFormSubmitButton"] button{
  border:none !important; border-radius:50% !important; width:42px; height:42px; min-height:42px; padding:0 !important;
  font-size:1.1rem; color:#fff !important; transition:transform .15s ease, box-shadow .15s ease;
}
.st-key-composer [data-testid="stHorizontalBlock"]{ flex-wrap:nowrap !important; gap:6px !important; }
.st-key-composer [data-testid="stColumn"]{ min-width:0 !important; }
.st-key-composer [data-testid="stColumn"]:nth-of-type(2), .st-key-composer [data-testid="stColumn"]:nth-of-type(3){
  flex:0 0 44px !important; width:44px !important; }
.st-key-composer [data-testid="stColumn"]:nth-of-type(2){ order:3; }  /* ➤ shown last */
.st-key-composer [data-testid="stColumn"]:nth-of-type(3){ order:2; }  /* ✨ shown before it */
.st-key-composer [data-testid="stColumn"]:nth-of-type(3) button{ background:linear-gradient(135deg,#8B5CF6,#EC4899) !important; box-shadow:0 6px 18px rgba(236,72,153,.4); }
.st-key-composer [data-testid="stColumn"]:nth-of-type(2) button{ background:linear-gradient(135deg,#14B8A6,#38BDF8) !important; box-shadow:0 6px 18px rgba(56,189,248,.35); }
.st-key-composer [data-testid="stFormSubmitButton"] button:hover{ transform:translateY(-1px) scale(1.05); }
.st-key-composer [data-testid="InputInstructions"]{ display:none; }
.st-key-composer [data-testid="stCaptionContainer"]{ padding-left:20px; margin-top:4px; }
.st-key-composer [data-testid="stCaptionContainer"] p{ color:#C4B5FD !important; }
@media (max-width:640px){ [data-testid="stChatMessage"]:has(.msg-user){ max-width:92%; } .hero h1{ font-size:1.9rem !important; } }
.st-key-composer .st-key-stopbtn button, .st-key-composer [data-testid="stColumn"] .st-key-stopbtn button{ background:linear-gradient(135deg,#F43F5E,#F97316) !important; box-shadow:0 6px 18px rgba(244,63,94,.45) !important; }
.typing{ display:inline-flex; gap:6px; align-items:center; padding:6px 2px; color:var(--muted); font-size:.95rem; }
.typing i{ width:8px; height:8px; border-radius:50%; background:linear-gradient(135deg,#A78BFA,#F472B6); display:inline-block;
  animation: blink 1.2s infinite ease-in-out; }
.typing i:nth-child(2){ animation-delay:.2s; } .typing i:nth-child(3){ animation-delay:.4s; }
@keyframes blink{ 0%,80%,100%{ opacity:.25; transform:translateY(0);} 40%{ opacity:1; transform:translateY(-3px);} }
/* ---------- job steps (live progress) ---------- */
.steps{ display:flex; flex-direction:column; gap:7px; margin:2px 0 4px; }
.steps .st{ display:grid; grid-template-columns:22px minmax(0,1fr); gap:8px; align-items:start; font-size:.95rem; color:var(--muted); }
.steps .st .ic{ text-align:center; line-height:1.45; }
.steps .st.done{ color:var(--ink); }
.steps .st.now{ color:#fff; font-weight:600; }
.steps .st.now .ic{ animation: blink 1.2s infinite ease-in-out; }
.steps .st .sub{ display:block; font-weight:400; font-size:.84rem; color:#C4B5FD; margin-top:1px; overflow-wrap:anywhere; }
@media (prefers-reduced-motion: reduce){ *{ transition:none !important; animation:none !important; } }
</style>
""")


# ============================== helpers ==============================
FRIENDLY_ERROR = "Sorry, I'm having trouble right now. Please try again in a moment."


@st.cache_resource
def _pool() -> ThreadPoolExecutor:
    return ThreadPoolExecutor(max_workers=4)


def md(text: str) -> str:
    """Escape $ so Streamlit doesn't render "$5M ... $10M" as a LaTeX formula."""
    return str(text).replace("\\$", "$").replace("$", "\\$")


def add_msg(role, content, kind="text", turn=None):
    ss.messages.append({"role": role, "content": content, "kind": kind,
                        "turn": ss.turn if turn is None else turn})


def history_for_llm():
    """The earlier chat Scout remembers (reports are shortened: their facts are kept in memory separately)."""
    return [{"role": m["role"], "content": m["content"][:600 if m.get("kind") == "report" else 900]}
            for m in ss.messages if m.get("kind") in ("text", "report")][-12:]


def typing_dots(ph, label="Scout is thinking"):
    ph.markdown(f'<span class="typing"><i></i><i></i><i></i>&nbsp;{label}</span>', unsafe_allow_html=True)


def wait_for(fn, ph, label="Scout is thinking"):
    """Run a slow call in a background thread while the page stays responsive.
    Updating the page every 0.25s gives Streamlit a chance to stop this run the moment the user
    presses ■ or sends another message."""
    fut = _pool().submit(fn)
    while not fut.done():
        typing_dots(ph, label)
        time.sleep(0.25)
    return fut.result()


def interrupt(note: str | None = None) -> bool:
    """The user stopped Scout (or sent a new message) while it was answering: undo that turn."""
    fl = ss.inflight
    if fl is None:
        return False
    ss.messages = [m for m in ss.messages if m["role"] == "user" or m.get("turn") != fl["turn"]]
    ss.conv_state = fl["state"]
    ss.pending, ss.job, ss.inflight = None, None, None
    if note:
        add_msg("assistant", note)
    return True


def iter_sse(response):
    """Minimal Server-Sent-Events parser. Keep-alive pings are yielded too, so the page can react to ■."""
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
            yield "ping", {}
        elif line.startswith("event:"):
            event = line[6:].strip()
        elif line.startswith("data:"):
            data_lines.append(line[5:].lstrip())


# ============================== "Download PDF" under every report ==============================
# The PDF is made in the user's browser (marked.js turns the report into a page, html2pdf.js saves it as a
# PDF), so it needs no extra Python packages and handles Arabic text correctly.
PDF_LIBS = ('<script src="https://cdn.jsdelivr.net/npm/marked@16.4.1/lib/marked.umd.js"></script>'
            '<script src="https://cdn.jsdelivr.net/npm/html2pdf.js@0.14.0/dist/html2pdf.bundle.min.js"></script>')
PDF_WIDGET = """<!doctype html><html><head><meta charset="utf-8">
<style>
  html,body{margin:0;background:transparent;font-family:"DM Sans","Segoe UI",system-ui,sans-serif}
  .row{display:flex;align-items:center;gap:10px;padding:4px 4px 10px}
  button{font:600 14px/1 inherit;font-family:inherit;color:#EAF0FF;cursor:pointer;padding:9px 16px;border-radius:999px;
    border:1.5px solid transparent;background:linear-gradient(#141a3a,#141a3a) padding-box,
    linear-gradient(100deg,#8B5CF6,#EC4899,#14B8A6) border-box;box-shadow:0 6px 18px rgba(139,92,246,.25)}
  button:hover{transform:translateY(-1px)} button:disabled{opacity:.6;cursor:default;transform:none}
  #msg{font-size:13px;color:#C4B5FD}
</style>__LIBS__</head><body>
<div class="row"><button id="b" type="button">⬇️ Download PDF</button><span id="msg"></span></div>
<script>
const REPORT = __MD__, FILE = __FILE__, DIR = __DIR__;
const CSS = `<style>
  .doc{width:700px;padding:6px 4px;color:#1b2233;background:#fff;font:12.5px/1.55 "Segoe UI",Arial,sans-serif}
  .doc .brand{background:linear-gradient(100deg,#8B5CF6,#EC4899);color:#fff;font-weight:700;font-size:12px;
    padding:8px 12px;border-radius:8px;margin-bottom:14px}
  .doc h1{font-size:22px;color:#4c1d95;margin:4px 0 2px} .doc h2{font-size:16px;color:#0f766e;margin:18px 0 6px;
    border-bottom:2px solid #ccfbf1;padding-bottom:3px} .doc h3{font-size:14px;color:#6d28d9;margin:16px 0 4px}
  .doc p{margin:5px 0} .doc ul{margin:4px 0;padding-inline-start:20px} .doc li{margin:2px 0}
  .doc a{color:#2563eb;text-decoration:none} .doc hr{border:none;border-top:1px solid #e5e7eb;margin:14px 0}
  .doc table{border-collapse:collapse;width:100%;margin:6px 0 10px;font-size:11.5px}
  .doc th{background:#ede9fe;color:#3b0764;text-align:start;padding:5px 7px}
  .doc td{border-bottom:1px solid #e5e7eb;padding:5px 7px;vertical-align:top;text-align:start}
  .doc em{color:#475569}
  .doc .newpage{page-break-before:always;break-before:page}
</style>`;
const b = document.getElementById("b"), msg = document.getElementById("msg");
b.onclick = async () => {
  if (!window.marked || !window.html2pdf) {
    msg.textContent = "Couldn't prepare the PDF. Check your internet connection and try again."; return;
  }
  b.disabled = true; msg.textContent = "Preparing your PDF…";
  try {
    const body = document.createElement("div");
    body.innerHTML = marked.parse(REPORT);
    // each company profile starts on a new page (the "Company profiles" heading moves with the first one)
    body.querySelectorAll("h3").forEach((h, i) => {
      const prev = h.previousElementSibling;
      const start = (i === 0 && prev && prev.tagName === "H2") ? prev : h;
      if (start !== body.firstElementChild) start.classList.add("newpage");
    });
    const page = `<div class="doc" dir="${DIR}">${CSS}<div class="brand">🔭 Scout · Arab Bank · Startup sourcing report</div>`
               + body.innerHTML + `</div>`;
    await html2pdf().set({
      margin: [12, 12, 14, 12], filename: FILE,
      image: {type: "jpeg", quality: 0.95},
      html2canvas: {scale: 2, backgroundColor: "#ffffff"},
      jsPDF: {unit: "mm", format: "a4", orientation: "portrait"},
      pagebreak: {mode: ["css", "legacy"], avoid: ["tr", "h1", "h2", "h3", "li", "p", "blockquote"]},
    }).from(page).save();
    msg.textContent = "Saved ✓";
  } catch (e) {
    console.error(e); msg.textContent = "Couldn't create the PDF. Please try again.";
  } finally { b.disabled = false; }
};
</script></body></html>"""


def _js_string(text: str) -> str:
    """A safe JavaScript string literal (can't close the <script> tag)."""
    return json.dumps(text).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")


def pdf_download(report: str):
    """Show the "Download PDF" link for one report."""
    first = next((ln for ln in report.splitlines() if ln.startswith("# ")), "# Scout report")
    title = re.sub(r'[\\/:*?"<>|#]+', "", first).strip()[:80] or "Scout report"
    arabic = sum("؀" <= c <= "ۿ" for c in report)
    letters = sum(c.isalpha() for c in report) or 1
    page = (PDF_WIDGET.replace("__LIBS__", PDF_LIBS)
            .replace("__MD__", _js_string(report.replace("<", "&lt;")))  # no raw HTML from the AI in the PDF
            .replace("__FILE__", _js_string(f"Scout - {title}.pdf"))
            .replace("__DIR__", _js_string("rtl" if arabic / letters > 0.3 else "ltr")))
    st.iframe(page, height=60)


# ============================== typing box callbacks ==============================
def on_send():
    text = (ss.draft or "").strip()
    if not text:
        return
    interrupt()  # a new message cancels whatever Scout was still doing (like any chat app)
    ss.turn += 1
    add_msg("user", text)
    ss.draft, ss.composer_note = "", ""
    if text.lower().strip(" !.") in ("try again", "retry", "again") and ss.failed_job:
        ss.job, ss.failed_job = ss.failed_job, None
        return
    ss.pending = {"message": text}


def on_stop():
    interrupt("⏹️ Stopped. What would you like to do instead?")


def on_enhance():
    text = (ss.draft or "").strip()
    if len(text.split()) < 2:
        ss.composer_note = "✨ Type your search idea first, then I'll polish it for you."
        return
    try:
        r = requests.post(f"{BACKEND_URL}/api/enhance", json={"criteria": text}, timeout=90)
        r.raise_for_status()
        ss.draft = r.json()["enhanced_criteria"]
        ss.composer_note = "✨ Improved your message. Edit it if you like, then press Enter to send."
    except Exception as e:  # noqa: BLE001
        log.warning("enhance failed: %s", e)
        ss.composer_note = "✨ I couldn't polish it right now. You can still send it as it is."


# ============================== header ==============================
st.markdown("""
<div class="hero"><div class="logo">🔭</div>
  <div><h1>Scout</h1><p>Arab Bank's startup scout: tell me what you're looking for and I'll research it for you.</p></div>
</div>
<div class="chips">
  <span class="chip v">🔬 <b>Deep research</b> + report</span>
  <span class="chip p">🗂️ <b>Quick lists</b> of names</span>
  <span class="chip t">🎯 <b>Company</b> lookups</span>
  <span class="chip a">⚡ <b>Startup</b> questions</span>
</div>
""", unsafe_allow_html=True)


def _post_chat(body):
    r = requests.post(f"{BACKEND_URL}/api/chat", json=body, timeout=180)
    r.raise_for_status()
    return r.json()


if not ss.messages:
    try:
        d = _post_chat({"action": "init", "state": {}, "history": []})
        ss.conv_state = d["state"]
        for m in d["messages"]:
            add_msg("assistant", m)
    except Exception as e:  # noqa: BLE001
        log.warning("backend not reachable: %s", e)
        add_msg("assistant", "Hi! 👋 I'm Scout. " + FRIENDLY_ERROR)

busy = ss.pending is not None or ss.job is not None

# ============================== typing box (fixed at the bottom) ==============================
with st.container(key="composer"):
    with st.form("composer_form", clear_on_submit=False, border=False, enter_to_submit=True):
        # ➤ comes BEFORE the other button in the page so that pressing Enter = Send (Streamlit submits
        # with the first button of a form). CSS (`order`) still shows it on the right.
        c_text, c_send, c_third = st.columns([14, 1.3, 1.3], vertical_alignment="center", gap="small")
        with c_text:
            st.text_input("Message", key="draft", label_visibility="collapsed",
                          placeholder=("Scout is working… type to send something else, or ■ to stop" if busy
                                       else "Ask Scout anything…  ✨ polishes your search idea"))
        with c_send:
            st.form_submit_button("➤", on_click=on_send, help="Send")
        with c_third:
            if busy:
                with st.container(key="stopbtn"):
                    st.form_submit_button("■", on_click=on_stop, help="Stop")
            else:
                st.form_submit_button("✨", on_click=on_enhance,
                                      help="Enhance with AI: rewrite my draft into a better search query")
    if ss.composer_note:
        st.caption(ss.composer_note)

# ============================== conversation ==============================
for m in ss.messages:
    is_user = m["role"] == "user"
    with st.chat_message(m["role"], avatar="🧑" if is_user else "🔭"):
        marker = "msg-user" if is_user else ("msg-report" if m["kind"] == "report" else "msg-bot")
        st.markdown(f'<span class="msg-marker {marker}"></span>', unsafe_allow_html=True)
        st.markdown(md(m["content"]))
        if m["kind"] == "report":
            pdf_download(m["content"])

# ============================== answer a pending message ==============================
if ss.pending is not None:
    payload, ss.pending = ss.pending, None
    ss.inflight = {"turn": ss.turn, "state": copy.deepcopy(ss.conv_state)}
    body = {"state": ss.conv_state, "history": history_for_llm()[:-1], **payload}
    with st.chat_message("assistant", avatar="🔭"):
        st.markdown('<span class="msg-marker msg-bot"></span>', unsafe_allow_html=True)
        try:
            d = wait_for(lambda: _post_chat(body), st.empty())
            ss.conv_state = d["state"]
            for m in d["messages"]:
                add_msg("assistant", m)
            if d.get("job"):
                ss.job = d["job"]          # keep `inflight` until the job finishes
            else:
                ss.inflight = None
        except Exception as e:  # noqa: BLE001
            log.warning("chat call failed: %s", e)
            add_msg("assistant", FRIENDLY_ERROR)
            ss.inflight = None
    st.rerun()


# ============================== run a job (live progress, then the result) ==============================
class JobSteps:
    """The live progress of a job, shown as a few clear steps:
         research:  🔎 Searching the market → 🏢 Found 3 startups → 🔬 Researching (1/3) → 📝 Writing your report
         list:      🔎 Searching the market → 🗂️ Making your list
         lookup:    🔬 Researching Tabby (0/1) → 📝 Writing your report
    Only the step in progress shows a small detail line (the latest search, the company being read…)."""

    PLANS = {
        "research": [("search", "🔎", "Searching the market"), ("found", "🏢", "Picking the best matches"),
                     ("research", "🔬", "Researching each startup"), ("report", "📝", "Writing your report")],
        "list": [("search", "🔎", "Searching the market"), ("found", "🗂️", "Making your list")],
        "lookup": [("research", "🔬", "Researching"), ("report", "📝", "Writing your report")],
    }

    def __init__(self, mode: str, status, progress):
        self.steps = [list(s) for s in self.PLANS.get(mode, self.PLANS["research"])]
        self.keys = [k for k, _, _ in self.steps]
        self.now, self.detail = 0, ""
        self.total, self.done, self.names = 0, 0, []
        self.status, self.progress = status, progress
        self.box = status.empty()

    def go(self, key: str, label: str | None = None):
        if key in self.keys:
            i = self.keys.index(key)
            self.now = max(self.now, i)
            if label:
                self.steps[i][2] = label
            self.detail = ""

    def research_label(self) -> str:
        n = self.names  # set for lookups ("Researching Tabby, Tamara"), empty after a market search
        who = (", ".join(n[:3]) + (" …" if len(n) > 3 else "")) if n else "each startup"
        return f"Researching {who} ({self.done}/{self.total} done)"

    def handle(self, event: str, data: dict):
        msg = data.get("message", "")
        if event == "status":
            self.go({"discovery": "search", "deep_dive": "research", "report": "report"}.get(data.get("stage"), ""))
            if data.get("stage") == "deep_dive":
                self.go("research", self.research_label())
        elif event == "search":
            self.detail = msg
        elif event == "discovered":
            names = [c["name"] for c in data.get("startups", [])]
            self.total = len(names)
            if data.get("stage") == "lookup":
                self.names = names
                self.go("research", self.research_label())
            else:
                n = len(names)
                self.go("found", f"Found {n} startup{'' if n == 1 else 's'}: {', '.join(names)}")
                if "research" in self.keys:
                    self.go("research", self.research_label())
                    self.now = self.keys.index("research")
        elif event in ("analyzing", "warning"):
            self.detail = msg
        elif event == "company_done":
            self.done += 1
            i = self.keys.index("research") if "research" in self.keys else None
            if i is not None:
                label = self.steps[i][2]
                self.steps[i][2] = re.sub(r"\(\d+/\d+ done\)", f"({self.done}/{self.total} done)", label)
            self.detail = msg

    def fraction(self, finished: bool = False) -> float:
        if finished:
            return 1.0
        part = 0.0
        if self.keys[self.now] == "research" and self.total:
            part = self.done / self.total
        return min(0.97, (self.now + part) / len(self.steps) + 0.03)

    def render(self, finished: bool = False):
        rows = []
        for i, (_, icon, label) in enumerate(self.steps):
            if finished or i < self.now:
                cls, ic = "done", "✅"
            elif i == self.now:
                cls, ic = "now", icon
            else:
                cls, ic = "", "○"
            sub = (f'<span class="sub">{html.escape(self.detail)}</span>'
                   if i == self.now and self.detail and not finished else "")
            rows.append(f'<div class="st {cls}"><span class="ic">{ic}</span>'
                        f'<span>{html.escape(label)}{sub}</span></div>')
        self.box.markdown('<div class="steps">' + "".join(rows) + "</div>", unsafe_allow_html=True)
        _, icon, label = self.steps[self.now]
        if not finished:
            self.status.update(label=f"{icon} {label}", expanded=True)  # keep the steps open while working
        self.progress.progress(self.fraction(finished))


def run_job(job: dict):
    final, problem = None, None
    with st.chat_message("assistant", avatar="🔭"):
        st.markdown('<span class="msg-marker msg-bot"></span>', unsafe_allow_html=True)
        status = st.status("Working on it…", expanded=True)
        progress = st.progress(0.0)
        steps = JobSteps(job["mode"], status, progress)
        steps.render()
        try:
            with requests.post(f"{BACKEND_URL}/api/search/stream", json=job,
                               stream=True, timeout=(10, 1800)) as resp:
                if resp.status_code != 200:
                    log.warning("job rejected: %s %s", resp.status_code, resp.text[:300])
                for event, data in iter_sse(resp):
                    if event == "ping":
                        progress.progress(steps.fraction())  # lets ■ stop the run between updates
                        continue
                    if event == "error":
                        log.warning("job error: %s", data.get("detail") or data.get("message"))
                        problem = data.get("message")
                        status.update(label=problem or FRIENDLY_ERROR, state="error")
                        break
                    if event == "complete":
                        final = data
                        steps.render(finished=True)
                        status.update(label="Done ✅", state="complete", expanded=False)
                        continue
                    steps.handle(event, data)
                    steps.render()
        except requests.exceptions.RequestException as e:
            log.warning("job connection failed: %s", e)

    if final:
        # only Deep Dive results are a "report" (with Download PDF); a list of names is a normal chat message
        add_msg("assistant", final.get("report") or FRIENDLY_ERROR,
                kind="text" if final.get("mode") == "list" else "report")
        state = dict(ss.conv_state)  # remember results so follow-up questions need no new searches
        new = final.get("memory", [])
        names = {m["name"].lower() for m in new}
        state["last_startups"] = (new + [m for m in state.get("last_startups", [])
                                         if m["name"].lower() not in names])[:15]
        state["last_batch"] = [m["name"] for m in new]  # what THIS job found ("just the names" lists these)
        if final.get("mode") != "lookup" or not state.get("last_criteria"):
            state["last_criteria"] = final.get("criteria")
        ss.conv_state = state
        ss.failed_job = None
        if final.get("mode") != "list":
            add_msg("assistant", "You can save the report with **Download PDF**. Ask me anything about these "
                                 "results, ask me to research one in more depth, or tell me what to look for next.")
    elif problem and "couldn't find" in problem.lower():
        add_msg("assistant", problem)  # nothing matched: trying the same search again won't help
    else:
        ss.failed_job = job
        add_msg("assistant", "Sorry, I couldn't finish that one. Say **try again**, or change your request.")


if ss.job is not None:
    if ss.inflight is None:
        ss.inflight = {"turn": ss.turn, "state": copy.deepcopy(ss.conv_state)}
    run_job(ss.job)
    ss.job, ss.inflight = None, None
    st.rerun()
