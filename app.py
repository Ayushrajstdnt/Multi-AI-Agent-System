"""
ResearchMind - Streamlit UI for the Multi-Agent Research System.

Run with:
    C:\\venvs\\multi-agent\\Scripts\\python.exe -m streamlit run app.py

Imports the agents from agents.py and runs the same four steps as pipeline.py
(search -> read -> write -> critique), with live status cards for each step.
"""

import html
import re
import time

import streamlit as st

st.set_page_config(page_title="ResearchMind", page_icon="🧠", layout="wide")

# ---------- Import the agents (readable error if setup is broken) ----------
try:
    from agents import (
        build_search_agent,
        build_reader_agent,
        writer_chain,
        critic_chain,
    )
except Exception as e:
    st.error(
        "Could not load the agents. Check that `.env` has `GROQ_API_KEY` and "
        "`TAVILY_API_KEY`, and that you started Streamlit with the venv Python."
    )
    st.exception(e)
    st.stop()


# ---------- Styling ----------
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=Space+Mono:wght@400;700&family=Lora:wght@400;500&family=Inter:wght@400;500;600&display=swap');

:root {
    --bg: #0a0a0d;
    --panel: #111116;
    --field: #1a1a20;
    --line: #25252d;
    --text: #e9e7e4;
    --muted: #8b8b95;
    --dim: #5d5d68;
    --orange: #ff8a2b;
    --orange-deep: #ff6a00;
    --green: #4ade80;
    --red: #f87171;
}

html, body, .stApp {
    background:
        radial-gradient(1100px 480px at 50% -8%, rgba(255,122,26,0.13), transparent 62%),
        var(--bg) !important;
    color: var(--text);
    color-scheme: dark;
    font-family: 'Inter', sans-serif;
}

#MainMenu, footer, header[data-testid="stHeader"] { visibility: hidden; height: 0; }
.block-container { max-width: 1080px; padding-top: 3rem; padding-bottom: 3rem; }

[data-testid="stMarkdownContainer"],
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li { color: #d6d3cf; }
[data-testid="stMarkdownContainer"] h1,
[data-testid="stMarkdownContainer"] h2,
[data-testid="stMarkdownContainer"] h3 { font-family: 'Syne', sans-serif; color: var(--text); }
[data-testid="stMarkdownContainer"] a { color: var(--orange); }

/* ----- Header ----- */
.rm-eyebrow {
    text-align: center; font-family: 'Space Mono', monospace; font-size: .72rem;
    letter-spacing: .3em; color: var(--orange);
}
.rm-title {
    text-align: center; font-family: 'Syne', sans-serif; font-weight: 800;
    font-size: clamp(1.4rem, 5.2vw, 4.2rem); letter-spacing: -0.03em; line-height: 1;
    white-space: nowrap; margin: 1rem 0 1.3rem; color: var(--text);
}
.rm-title span { color: var(--orange); }
.rm-sub {
    text-align: center; font-family: 'Lora', serif; color: #c9c6c2;
    max-width: 540px; margin: 0 auto; line-height: 1.7; font-size: 1rem;
}
.rm-rule { border: 0; border-top: 1px solid var(--line); margin: 2.2rem auto; max-width: 780px; }

/* ----- Left column: input ----- */
.rm-label {
    font-family: 'Space Mono', monospace; font-size: .72rem; letter-spacing: .14em;
    color: var(--orange); margin: .2rem 0 .45rem;
}
div[data-baseweb="input"] {
    background: var(--field) !important; border: 1px solid var(--line) !important;
    border-radius: 8px !important;
}
div[data-baseweb="input"]:focus-within {
    border-color: var(--orange) !important; box-shadow: 0 0 0 1px var(--orange) !important;
}
div[data-baseweb="base-input"] { background: transparent !important; }
div[data-testid="stTextInput"] input {
    color: var(--text) !important; background: transparent !important; padding: .7rem .9rem;
}
div[data-testid="stTextInput"] input::placeholder { color: var(--dim); }

button[data-testid="stBaseButton-primary"], button[kind="primary"] {
    background: linear-gradient(90deg, #ff8a2b, #ff6a00) !important;
    border: none !important; color: #1a0d00 !important; font-weight: 600;
    border-radius: 8px; width: 100%; padding: .65rem 1rem;
    box-shadow: 0 6px 26px rgba(255,122,26,.28);
}
button[data-testid="stBaseButton-primary"]:hover { filter: brightness(1.08); }

.rm-try {
    font-family: 'Space Mono', monospace; font-size: .68rem; letter-spacing: .12em;
    color: var(--dim); margin: 1.4rem 0 .4rem;
}
button[data-testid="stBaseButton-secondary"], button[kind="secondary"] {
    background: #14141a !important; border: 1px solid var(--line) !important;
    color: #c9c6c2 !important; font-family: 'Space Mono', monospace; font-size: .74rem;
    padding: .2rem .7rem; min-height: 0; border-radius: 6px;
}
button[data-testid="stBaseButton-secondary"]:hover { border-color: var(--orange) !important; color: var(--orange) !important; }

/* ----- Right column: pipeline cards ----- */
.rm-h { font-family: 'Syne', sans-serif; font-weight: 700; font-size: 1.35rem; margin: 0 0 1rem; color: var(--text); }
.step {
    background: var(--panel); border: 1px solid var(--line); border-radius: 10px;
    padding: 1rem 1.15rem; margin-bottom: .8rem;
    transition: border-color .2s, box-shadow .2s;
}
.step-top { display: flex; align-items: center; gap: .75rem; }
.step .num { font-family: 'Space Mono', monospace; font-size: .7rem; color: var(--muted); }
.step .name { font-family: 'Syne', sans-serif; font-weight: 700; font-size: 1.02rem; color: var(--text); flex: 1; }
.step .badge { font-family: 'Space Mono', monospace; font-size: .66rem; letter-spacing: .14em; color: var(--dim); }
.step .desc { font-size: .8rem; color: var(--muted); margin-top: .35rem; padding-left: 1.65rem; }
.step.running { border-color: var(--orange); box-shadow: 0 0 0 1px rgba(255,138,43,.35), 0 0 28px rgba(255,122,26,.16); }
.step.running .badge { color: var(--orange); animation: rm-pulse 1.2s ease-in-out infinite; }
.step.done { border-color: rgba(74,222,128,.35); }
.step.done .badge { color: var(--green); }
.step.error { border-color: rgba(248,113,113,.55); }
.step.error .badge { color: var(--red); }
@keyframes rm-pulse { 50% { opacity: .35; } }
@media (prefers-reduced-motion: reduce) { .step.running .badge { animation: none; } }

/* ----- Results ----- */
.rm-stats { display: grid; grid-template-columns: repeat(3, 1fr); gap: .8rem; margin: 1rem 0 1.6rem; }
.rm-stat { background: var(--panel); border: 1px solid var(--line); border-radius: 10px; padding: .9rem 1.1rem; }
.rm-stat b { font-family: 'Syne', sans-serif; font-size: 1.8rem; display: block; color: var(--orange); line-height: 1.2; }
.rm-stat span { font-family: 'Space Mono', monospace; font-size: .68rem; color: var(--muted); letter-spacing: .08em; }

button[data-baseweb="tab"] { font-family: 'Space Mono', monospace; font-size: .78rem; color: var(--muted); }
button[data-baseweb="tab"][aria-selected="true"] { color: var(--orange); }
div[data-baseweb="tab-highlight"] { background: var(--orange) !important; }
div[data-baseweb="tab-border"] { background: var(--line) !important; }

.rm-foot {
    text-align: center; font-family: 'Space Mono', monospace; font-size: .68rem;
    color: var(--dim); letter-spacing: .08em; margin-top: 3.5rem;
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ---------- Config ----------
STEPS = [
    ("search", "Search Agent", "Gathers recent web information"),
    ("read", "Reader Agent", "Scrapes & extracts deep content"),
    ("write", "Writer Chain", "Drafts the full research report"),
    ("review", "Critic Chain", "Reviews & scores the report"),
]

EXAMPLE_TOPICS = [
    "Iran war impact on oil prices",
    "LLM agents in 2026",
    "Data analyst skills in India 2026",
]

URL_PATTERN = re.compile(r"https?://[^\s\)\]>\"']+")


# ---------- Helpers ----------
def as_text(content) -> str:
    """Some models return a list of content blocks instead of a string."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            parts.append(block.get("text", "") if isinstance(block, dict) else str(block))
        return "\n".join(p for p in parts if p)
    return str(content)


def extract_score(feedback: str):
    match = re.search(r"Score:\s*(\d+(?:\.\d+)?)\s*/\s*10", feedback)
    return match.group(1) if match else None


def extract_urls(text: str):
    seen, urls = set(), []
    for url in URL_PATTERN.findall(text):
        url = url.rstrip(".,;")
        if url not in seen:
            seen.add(url)
            urls.append(url)
    return urls


def friendly_error(err: Exception) -> str:
    msg = str(err)
    if "429" in msg or "rate limit" in msg.lower():
        return "Groq's free-tier rate limit was hit. Wait a minute and run it again."
    if "model_not_found" in msg or "does not exist" in msg:
        return "The model in `agents.py` isn't available on your Groq account. Pick a different model."
    if "tool_use_failed" in msg or "tool call validation failed" in msg:
        return "The model called a tool that doesn't exist. Run it again; this is intermittent."
    if "api_key" in msg.lower() or "401" in msg:
        return "An API key is missing or invalid. Check `GROQ_API_KEY` and `TAVILY_API_KEY` in `.env`."
    return "Something went wrong while running the pipeline."


def render_pipeline(slot, states: dict, timings: dict):
    """Draws the four step cards and remembers them for the next rerun."""
    st.session_state["pipe_states"] = dict(states)
    st.session_state["pipe_times"] = dict(timings)

    labels = {"waiting": "WAITING", "running": "RUNNING", "done": "DONE", "error": "FAILED"}
    cards = []
    for i, (key, name, desc) in enumerate(STEPS, start=1):
        state = states.get(key, "waiting")
        label = labels[state]
        if state == "done" and key in timings:
            label = f"DONE · {timings[key]:.0f}s"
        cards.append(
            f'<div class="step {state}"><div class="step-top">'
            f'<span class="num">{i:02d}</span><span class="name">{name}</span>'
            f'<span class="badge">{label}</span></div>'
            f'<div class="desc">{desc}</div></div>'
        )
    slot.markdown("".join(cards), unsafe_allow_html=True)


def run_pipeline(topic: str, slot) -> dict:
    states = {key: "waiting" for key, _, _ in STEPS}
    timings = {}
    result = {"topic": topic}
    current = None

    def begin(key):
        nonlocal current
        current = key
        states[key] = "running"
        render_pipeline(slot, states, timings)
        return time.time()

    def finish(key, started):
        states[key] = "done"
        timings[key] = time.time() - started
        render_pipeline(slot, states, timings)

    render_pipeline(slot, states, timings)

    try:
        # 1. Search
        t = begin("search")
        search_agent = build_search_agent()
        out = search_agent.invoke(
            {"messages": [("user", f"Find recent, reliable and detailed information about: {topic}")]}
        )
        result["search_results"] = as_text(out["messages"][-1].content)
        finish("search", t)

        # 2. Read
        t = begin("read")
        reader_agent = build_reader_agent()
        out = reader_agent.invoke(
            {
                "messages": [
                    (
                        "user",
                        f"Based on the following search results about '{topic}', "
                        f"pick the most relevant URL and scrape it for deeper content.\n\n"
                        f"Search Results:\n{result['search_results'][:2000]}",
                    )
                ]
            }
        )
        result["scraped_content"] = as_text(out["messages"][-1].content)
        finish("read", t)

        # 3. Write
        t = begin("write")
        research = (
            f"SEARCH RESULTS:\n{result['search_results'][:2000]}\n\n"
            f"DETAILED SCRAPED CONTENT:\n{result['scraped_content'][:4000]}"
        )
        result["report"] = as_text(writer_chain.invoke({"topic": topic, "research": research}))
        finish("write", t)

        # 4. Review
        t = begin("review")
        result["feedback"] = as_text(critic_chain.invoke({"report": result["report"]}))
        finish("review", t)

    except Exception:
        if current:
            states[current] = "error"
            render_pipeline(slot, states, timings)
        raise

    result["timings"] = timings
    return result


def set_topic(value: str):
    st.session_state["topic_input"] = value


# ---------- Header ----------
st.markdown(
    '<div class="rm-eyebrow">MULTI-AGENT AI SYSTEM</div>'
    '<div class="rm-title">Research<span>Mind</span></div>'
    '<div class="rm-sub">Four specialized AI agents collaborate — searching, scraping, '
    "writing, and critiquing — to deliver a polished research report on any topic.</div>"
    '<hr class="rm-rule">',
    unsafe_allow_html=True,
)

# ---------- Main two-column layout ----------
left, right = st.columns([1, 1], gap="large")

with left:
    st.markdown('<div class="rm-label">RESEARCH TOPIC</div>', unsafe_allow_html=True)
    topic = st.text_input(
        "Research topic",
        key="topic_input",
        placeholder="e.g. Quantum computing breakthroughs in 2026",
        label_visibility="collapsed",
    )
    run_clicked = st.button("⚡  Run Research Pipeline", type="primary")

    st.markdown('<div class="rm-try">TRY →</div>', unsafe_allow_html=True)
    for example in EXAMPLE_TOPICS:
        st.button(example, key=f"ex_{example}", on_click=set_topic, args=(example,))

with right:
    st.markdown('<div class="rm-h">Pipeline</div>', unsafe_allow_html=True)
    pipe_slot = st.empty()
    render_pipeline(
        pipe_slot,
        st.session_state.get("pipe_states", {}),
        st.session_state.get("pipe_times", {}),
    )

# ---------- Run ----------
if run_clicked:
    if not topic.strip():
        st.warning("Enter a research topic first.")
    else:
        st.session_state.pop("result", None)
        try:
            st.session_state["result"] = run_pipeline(topic.strip(), pipe_slot)
        except Exception as err:
            st.error(friendly_error(err))
            with st.expander("Technical details"):
                st.code(str(err))

# ---------- Results ----------
result = st.session_state.get("result")

if result:
    st.markdown('<hr class="rm-rule">', unsafe_allow_html=True)
    st.markdown(f'<div class="rm-h">{html.escape(result["topic"])}</div>', unsafe_allow_html=True)

    score = extract_score(result["feedback"])
    n_sources = len(extract_urls(result["search_results"]))
    total_time = sum(result["timings"].values())

    st.markdown(
        '<div class="rm-stats">'
        f'<div class="rm-stat"><b>{score + "/10" if score else "n/a"}</b><span>CRITIC SCORE</span></div>'
        f'<div class="rm-stat"><b>{n_sources}</b><span>SOURCES FOUND</span></div>'
        f'<div class="rm-stat"><b>{total_time:.0f}s</b><span>TOTAL TIME</span></div>'
        "</div>",
        unsafe_allow_html=True,
    )

    tab_report, tab_critic, tab_sources, tab_raw = st.tabs(
        ["Report", "Critic feedback", "Sources", "Raw research"]
    )

    with tab_report:
        st.markdown(result["report"])
        st.download_button(
            "Download report (.md)",
            data=result["report"],
            file_name="research_report.md",
            mime="text/markdown",
        )

    with tab_critic:
        st.markdown(result["feedback"])

    with tab_sources:
        urls = extract_urls(result["search_results"])
        if urls:
            for url in urls:
                st.markdown(f"- {url}")
        else:
            st.info("No URLs were found in the search results.")

    with tab_raw:
        st.markdown("**Search results**")
        st.text(result["search_results"])
        st.markdown("**Scraped content**")
        st.text(result["scraped_content"])

st.markdown(
    '<div class="rm-foot">ResearchMind · Powered by LangChain multi-agent pipeline · Built with Streamlit</div>',
    unsafe_allow_html=True,
)