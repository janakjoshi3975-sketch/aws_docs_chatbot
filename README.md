# AWS Docs Agent

An agentic chatbot that lets you explore AWS through natural language. The agent
searches and reads the official AWS documentation (`docs.aws.amazon.com`) to
ground every answer it gives. It's "agentic" in the literal sense: it decides
which tools to use, in what order, and when it has enough information.

Built with the Anthropic Python SDK + Streamlit.

---

## Quick start

```bash
# 1. Clone / cd into the project, then:
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Set your Anthropic key
export ANTHROPIC_API_KEY=sk-ant-...
# or: cp .env.example .env  and edit it (then `source .env` or use python-dotenv)

# 3a. Run the chat UI
streamlit run app.py

# 3b. Or use the CLI
python cli.py
python cli.py "What is the max object size in S3?"   # one-shot
```

> **Note:** Web search must be enabled for your Anthropic organization in the
> Console (`Settings → Privacy → Web search`). Web fetch is currently behind a
> beta header, which the agent already passes.

---

## Architecture

```
┌────────────────┐       ┌───────────────────────┐
│ app.py / cli.py│──────▶│   AWSDocsAgent.chat() │
│  (UI / loop)   │       │   ─ tool-use loop ─   │
└────────────────┘       └─────────┬─────────────┘
                                   │
                  ┌────────────────┼─────────────────┐
                  ▼                ▼                 ▼
        ┌──────────────────┐  ┌──────────┐   ┌────────────────┐
        │  web_search      │  │ web_fetch│   │ save_research_ │
        │  (server tool,   │  │ (server  │   │ note (client   │
        │  AWS docs only)  │  │ tool)    │   │ tool, local)   │
        └──────────────────┘  └──────────┘   └────────────────┘
                  └─── Anthropic ─┘             └─ your code ─┘
```

The loop in `agent.py` does this on each user message:

1. Call `messages.create(...)` with the conversation + tool specs.
2. The model returns a list of content blocks: text, server tool calls
   (already executed by Anthropic, results inline), and/or client tool calls.
3. If the model called any **client** tools, run them locally and append the
   results as a new user turn, then loop back to step 1.
4. When the model stops without calling a client tool, we're done.

`web_search` and `web_fetch` are *server* tools — they run on Anthropic's side
and the SDK returns their results inline, so the loop only needs to handle
client tools explicitly. Both are domain-restricted to `docs.aws.amazon.com`.

---

## Files

| File | Purpose |
|---|---|
| `agent.py` | Core `AWSDocsAgent` class, tool specs, agent loop, event types |
| `app.py` | Streamlit chat UI |
| `cli.py` | Terminal CLI (interactive or one-shot) |
| `requirements.txt` | `anthropic`, `streamlit`, `python-dotenv` |
| `aws_research_notes.md` | Auto-created when the agent uses `save_research_note` |

---

## Things to ask it

- _Compare Aurora Serverless v2 and DynamoDB on-demand for a write-heavy workload._
- _What IAM permissions does `aws s3 sync` need?_
- _How does SageMaker Pipelines pass artifacts between steps?_
- _What's the cold-start behavior of Lambda SnapStart for Python?_
- _Save a note summarizing VPC endpoint pricing._
- _What are the rate limits for Bedrock Claude models?_

---

## Notes for data scientists

- The agent picks `claude-opus-4-7` by default (best reasoning); switch to
  Sonnet/Haiku in the sidebar if you want lower latency/cost.
- All conversation history is in `st.session_state.history` — easy to dump
  to a file or notebook for analysis.
- Token usage is shown after each turn; the SDK returns it on the response
  object so you can log it.
- The `save_research_note` tool is just an example — repoint it at a SQLite
  DB or a pandas DataFrame if you want structured logs of what the agent
  has looked up.
