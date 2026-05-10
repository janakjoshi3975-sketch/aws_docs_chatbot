"""
Streamlit chat UI for the AWS Documentation Agent.

Run:
    streamlit run app.py
"""

from __future__ import annotations

import os
from pathlib import Path

import streamlit as st

from agent import (
    AWSDocsAgent,
    ClientToolEvent,
    FinalEvent,
    ServerToolEvent,
    ServerToolResultEvent,
    TextEvent,
    DEFAULT_MODEL,
    DEFAULT_NOTES_PATH,
    SYSTEM_PROMPT,
)


# ------------------------------------------------------------ page setup

st.set_page_config(
    page_title="AWS Docs Agent",
    page_icon="📚",
    layout="centered",
)

st.markdown(
    """
    <style>
    .tool-badge {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 10px;
        background: #232f3e;
        color: #ff9900;
        font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
        font-size: 12px;
        margin-right: 6px;
    }
    .tool-result {
        font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
        font-size: 12px;
        color: #555;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("📚 AWS Docs Agent")
st.caption(
    "Ask anything about AWS. The agent will search and read the official docs "
    "(`docs.aws.amazon.com`) before answering."
)


# ------------------------------------------------------------ sidebar

with st.sidebar:
    st.header("Settings")

    api_key_present = bool(os.environ.get("ANTHROPIC_API_KEY"))
    if not api_key_present:
        st.error("`ANTHROPIC_API_KEY` is not set in the environment.")
    else:
        st.success("API key detected.")

    model = st.selectbox(
        "Model",
        options=[
            "claude-opus-4-7",
            "claude-opus-4-6",
            "claude-sonnet-4-6",
            "claude-haiku-4-5",
        ],
        index=0,
        help="Opus is best for multi-step reasoning; Sonnet/Haiku are faster.",
    )
    max_tokens = st.slider("Max output tokens", 512, 8192, 4096, step=512)
    max_iterations = st.slider(
        "Max agent iterations", 1, 20, 10,
        help="Hard cap on tool-use loops per user turn.",
    )

    notes_path = st.text_input(
        "Notes file path",
        value=str(Path(DEFAULT_NOTES_PATH).resolve()),
        help="Where `save_research_note` writes. Tell the agent to save a note to populate it.",
    )

    with st.expander("System prompt"):
        system_prompt = st.text_area(
            "Edit the agent's instructions",
            value=SYSTEM_PROMPT,
            height=300,
        )

    st.divider()
    if st.button("🧹 Reset chat", use_container_width=True):
        st.session_state.history = []
        st.session_state.display = []
        st.rerun()

    st.divider()
    st.markdown(
        "**Try asking**\n"
        "- _Compare Aurora Serverless v2 vs DynamoDB on-demand for a write-heavy workload_\n"
        "- _What IAM permissions does `aws s3 sync` need?_\n"
        "- _How does SageMaker Pipelines pass artifacts between steps?_\n"
        "- _Save a note about VPC endpoint pricing_\n"
    )


# ------------------------------------------------------------ session state

if "history" not in st.session_state:
    # Raw message history passed to the API. Owned by us (the UI).
    st.session_state.history = []

if "display" not in st.session_state:
    # Display log: list of dicts {role, text, events}. Separate from API history
    # so we can render tool calls inline as expanders.
    st.session_state.display = []


# ------------------------------------------------------------ render history

def render_events(events: list) -> None:
    """Render tool-use events as compact expanders."""
    for e in events:
        if isinstance(e, ServerToolEvent):
            label = "🔍 search" if e.name == "web_search" else "📄 fetch"
            q = e.input.get("query") or e.input.get("url") or str(e.input)
            with st.expander(f"{label} · `{q}`", expanded=False):
                st.json(e.input)
        elif isinstance(e, ServerToolResultEvent):
            st.markdown(
                f"<div class='tool-result'>↳ {e.summary}</div>",
                unsafe_allow_html=True,
            )
        elif isinstance(e, ClientToolEvent):
            with st.expander(f"🛠️ {e.name}", expanded=False):
                st.json({"input": e.input, "result": e.result})


for turn in st.session_state.display:
    with st.chat_message(turn["role"]):
        if turn["role"] == "assistant" and turn.get("events"):
            render_events(turn["events"])
        st.markdown(turn["text"])


# ------------------------------------------------------------ chat input

prompt = st.chat_input(
    "Ask about an AWS service, limit, IAM action, architecture pattern…",
    disabled=not api_key_present,
)

if prompt:
    # Show user message immediately
    st.session_state.display.append({"role": "user", "text": prompt, "events": []})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Build the agent with current sidebar settings
    agent = AWSDocsAgent(
        model=model,
        max_tokens=max_tokens,
        system_prompt=system_prompt,
        notes_path=notes_path,
        max_iterations=max_iterations,
    )

    st.session_state.history.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        events_placeholder = st.container()
        text_placeholder = st.empty()
        text_buffer: list[str] = []
        events_collected: list = []

        with st.spinner("Thinking…"):
            try:
                for event in agent.chat(st.session_state.history):
                    events_collected.append(event)
                    if isinstance(event, TextEvent):
                        text_buffer.append(event.text)
                        text_placeholder.markdown("\n".join(text_buffer))
                    elif isinstance(event, (ServerToolEvent, ServerToolResultEvent, ClientToolEvent)):
                        with events_placeholder:
                            render_events([event])
                    elif isinstance(event, FinalEvent):
                        # Optional: show usage in a caption
                        usage = event.usage or {}
                        if usage:
                            st.caption(
                                f"stop: `{event.stop_reason}` · "
                                f"in: {usage.get('input_tokens', '?')} · "
                                f"out: {usage.get('output_tokens', '?')}"
                            )
            except Exception as exc:  # noqa: BLE001
                st.error(f"Agent error: {exc!r}")
                text_buffer.append(f"_(error: {exc!r})_")

        final_text = "\n".join(text_buffer).strip() or "_(no text response)_"
        # Don't double-write the text — placeholder already has it.

    st.session_state.display.append(
        {
            "role": "assistant",
            "text": final_text,
            "events": [
                e for e in events_collected
                if not isinstance(e, (TextEvent, FinalEvent))
            ],
        }
    )
