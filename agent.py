"""
AWS Documentation Chatbot — core agent.

The agent is "agentic" in the literal sense: it decides which tools to use,
in what order, and when it has enough information to answer. The orchestration
is a standard tool-use loop:

    user message
      -> model produces (text | tool_use)*
      -> if tool_use, run the tool, return tool_result, loop
      -> when model returns end_turn with no tool_use, we're done

Two flavors of tools are wired up:

1. SERVER TOOLS (Anthropic-executed, restricted to docs.aws.amazon.com):
     - web_search:  search AWS docs
     - web_fetch:   load a specific AWS doc page
   These run on Anthropic's side; the SDK handles them transparently and
   they appear in the response as `server_tool_use` / tool_result blocks
   that we just need to display.

2. CLIENT TOOLS (we execute them locally):
     - save_research_note:  append a finding to a local markdown notebook
   This demonstrates how to extend the agent with your own Python functions
   (e.g. query a vector store, hit your AWS account, log to a DB, etc.).

To extend: add a tool spec to CLIENT_TOOLS and a handler in run_client_tool().
"""

from __future__ import annotations

import json
import os
import datetime as dt
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Callable

import anthropic
from dotenv import load_dotenv
load_dotenv()


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

DEFAULT_MODEL = "claude-opus-4-7"
DEFAULT_MAX_TOKENS = 4096
DEFAULT_NOTES_PATH = Path("./aws_research_notes.md")

ALLOWED_DOMAINS = ["docs.aws.amazon.com"]

SYSTEM_PROMPT = """\
You are an expert AWS solutions architect and documentation guide. Your job is to
help the user understand AWS services, find best practices, and design solutions
by grounding every answer in the official AWS documentation.

You have these tools:
  - web_search: search the AWS documentation site
  - web_fetch:  retrieve the full content of a specific AWS docs page
  - save_research_note: append a finding to the user's local notebook

How to work:
  1. For any factual claim about an AWS service (limits, behavior, pricing
     model, supported features, IAM actions, etc.), search or fetch the docs
     before answering. Don't rely on memory for specifics.
  2. Prefer searching first to find candidate pages, then fetch the most
     promising one for detail.
  3. For comparison or architecture questions, search for each service
     involved and synthesize.
  4. Always include source URLs from the docs when you make a claim.
  5. When the user explicitly asks you to "save", "note", "remember", or
     "log" a finding, call save_research_note.
  6. Be concise. Use short paragraphs and bullet lists for trade-offs.
  7. If a question is genuinely outside AWS documentation (e.g. opinions,
     non-AWS comparisons), say so plainly and answer from general knowledge,
     marking it as not from docs.
"""


# -----------------------------------------------------------------------------
# Tool specs
# -----------------------------------------------------------------------------

def server_tools() -> list[dict[str, Any]]:
    """Anthropic-executed tools, restricted to AWS docs."""
    return [
        {
            "type": "web_search_20250305",
            "name": "web_search",
            "max_uses": 8,
            "allowed_domains": ALLOWED_DOMAINS,
        },
        {
            "type": "web_fetch_20250910",
            "name": "web_fetch",
            "max_uses": 5,
            "allowed_domains": ALLOWED_DOMAINS,
            "citations": {"enabled": True},
        },
    ]


CLIENT_TOOLS: list[dict[str, Any]] = [
    {
        "name": "save_research_note",
        "description": (
            "Append a research note to the user's local markdown notebook. "
            "Use this when the user asks to save, note, log, or remember "
            "something you've found, or when you discover a particularly "
            "useful fact worth keeping. The note is timestamped automatically."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "topic": {
                    "type": "string",
                    "description": "Short topic line, e.g. 'S3 multipart upload limits'.",
                },
                "summary": {
                    "type": "string",
                    "description": "Markdown body of the note. Include source URLs.",
                },
            },
            "required": ["topic", "summary"],
        },
    },
]


# -----------------------------------------------------------------------------
# Client tool implementations
# -----------------------------------------------------------------------------

def _save_research_note(
    topic: str,
    summary: str,
    notes_path: Path = DEFAULT_NOTES_PATH,
) -> str:
    notes_path = Path(notes_path)
    notes_path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    entry = f"\n## {topic}\n_{timestamp}_\n\n{summary}\n"
    with notes_path.open("a", encoding="utf-8") as f:
        f.write(entry)
    return f"Saved note '{topic}' to {notes_path.resolve()}."


CLIENT_TOOL_HANDLERS: dict[str, Callable[..., str]] = {
    "save_research_note": _save_research_note,
}


def run_client_tool(name: str, tool_input: dict[str, Any]) -> str:
    """Dispatch a client-side tool call. Returns a string result."""
    handler = CLIENT_TOOL_HANDLERS.get(name)
    if handler is None:
        return f"Error: unknown client tool '{name}'."
    try:
        return handler(**tool_input)
    except Exception as exc:  # noqa: BLE001 - we want to surface the error to the model
        return f"Error running {name}: {exc!r}"


# -----------------------------------------------------------------------------
# Event types yielded by the agent (UI-friendly)
# -----------------------------------------------------------------------------

@dataclass
class TextEvent:
    text: str

@dataclass
class ServerToolEvent:
    name: str          # 'web_search' | 'web_fetch'
    input: dict[str, Any]

@dataclass
class ServerToolResultEvent:
    name: str
    summary: str       # short human-readable summary, e.g. "5 results"

@dataclass
class ClientToolEvent:
    name: str
    input: dict[str, Any]
    result: str

@dataclass
class FinalEvent:
    stop_reason: str
    usage: dict[str, Any] = field(default_factory=dict)

AgentEvent = TextEvent | ServerToolEvent | ServerToolResultEvent | ClientToolEvent | FinalEvent


# -----------------------------------------------------------------------------
# The agent
# -----------------------------------------------------------------------------

class AWSDocsAgent:
    """
    Stateless wrapper around the Anthropic Messages API that runs the
    tool-use loop until the model is done.

    Conversation history is owned by the caller (the UI) — this class just
    takes a list of messages and yields events for the latest turn.
    """

    def __init__(
        self,
        client: anthropic.Anthropic | None = None,
        model: str = DEFAULT_MODEL,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        system_prompt: str = SYSTEM_PROMPT,
        notes_path: Path | str = DEFAULT_NOTES_PATH,
        max_iterations: int = 10,
    ) -> None:
        self.client = client or anthropic.Anthropic()
        self.model = model
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt
        self.notes_path = Path(notes_path)
        self.max_iterations = max_iterations

    # ------------------------------------------------------------------ utils

    def _tools(self) -> list[dict[str, Any]]:
        return server_tools() + CLIENT_TOOLS

    def _run_client_tool(self, name: str, tool_input: dict[str, Any]) -> str:
        if name == "save_research_note":
            return _save_research_note(
                topic=tool_input.get("topic", ""),
                summary=tool_input.get("summary", ""),
                notes_path=self.notes_path,
            )
        return run_client_tool(name, tool_input)

    # --------------------------------------------------------------- the loop

    def chat(self, history: list[dict[str, Any]]) -> Iterable[AgentEvent]:
        """
        Run the agent on a conversation history (list of {role, content} dicts).
        Yields AgentEvents as the response is built. The final event is FinalEvent.

        The caller is responsible for appending the assistant's final
        `content` blocks back onto `history` after the generator finishes,
        if they want multi-turn continuity. (See `chat_and_collect` for a
        helper that does it.)
        """
        # We need to mutate `history` in-place across iterations because tool
        # results have to be appended as new user messages. To avoid surprising
        # the caller we take a shallow copy.
        messages: list[dict[str, Any]] = list(history)

        for _ in range(self.max_iterations):
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=self.system_prompt,
                tools=self._tools(),
                messages=messages,
                extra_headers={"anthropic-beta": "web-fetch-2025-09-10"},
            )

            # Append the assistant turn (with all blocks, including tool_use
            # and any server_tool_use blocks) to messages, so subsequent
            # iterations have the full context.
            assistant_blocks = [self._block_to_dict(b) for b in response.content]
            messages.append({"role": "assistant", "content": assistant_blocks})

            # Walk the blocks and yield events.
            client_tool_uses: list[Any] = []
            for block in response.content:
                btype = getattr(block, "type", None)

                if btype == "text":
                    yield TextEvent(text=block.text)

                elif btype == "server_tool_use":
                    yield ServerToolEvent(
                        name=block.name,
                        input=getattr(block, "input", {}) or {},
                    )

                elif btype in ("web_search_tool_result", "web_fetch_tool_result"):
                    yield ServerToolResultEvent(
                        name=btype.replace("_tool_result", ""),
                        summary=self._summarize_server_result(block),
                    )

                elif btype == "tool_use":
                    # Client tool — we have to run it.
                    client_tool_uses.append(block)

            # If the model used a client tool, run it and feed the result back.
            if client_tool_uses:
                tool_result_blocks = []
                for tu in client_tool_uses:
                    result = self._run_client_tool(tu.name, tu.input or {})
                    yield ClientToolEvent(
                        name=tu.name,
                        input=tu.input or {},
                        result=result,
                    )
                    tool_result_blocks.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": tu.id,
                            "content": result,
                        }
                    )
                messages.append({"role": "user", "content": tool_result_blocks})
                # Continue the loop — the model will see the results and respond.
                continue

            # No client tool calls and we're at end_turn (or any non-tool stop) —
            # the agent is done.
            yield FinalEvent(
                stop_reason=response.stop_reason or "end_turn",
                usage=dict(response.usage) if hasattr(response, "usage") else {},
            )
            # IMPORTANT: write the assistant blocks back to the caller's history
            # so they can pass it into the next turn.
            history.clear()
            history.extend(messages)
            return

        # Safety net: ran out of iterations.
        yield FinalEvent(stop_reason="max_iterations", usage={})
        history.clear()
        history.extend(messages)

    # ------------------------------------------------------------- formatting

    @staticmethod
    def _block_to_dict(block: Any) -> dict[str, Any]:
        """Convert SDK block objects back to plain dicts for the next API call."""
        if hasattr(block, "model_dump"):
            return block.model_dump(exclude_none=True)
        if hasattr(block, "dict"):
            return block.dict(exclude_none=True)
        return dict(block)  # last resort

    @staticmethod
    def _summarize_server_result(block: Any) -> str:
        """One-line human summary of a server tool result block."""
        content = getattr(block, "content", None)
        if content is None:
            return "(no result)"
        # web_search returns a list of result objects; web_fetch returns one.
        if isinstance(content, list):
            urls = []
            for item in content:
                url = getattr(item, "url", None) or (
                    item.get("url") if isinstance(item, dict) else None
                )
                if url:
                    urls.append(url)
            if urls:
                return f"{len(urls)} result(s): " + ", ".join(urls[:3]) + (
                    " ..." if len(urls) > 3 else ""
                )
            return f"{len(content)} result(s)"
        # web_fetch single result
        url = getattr(content, "url", None) or (
            content.get("url") if isinstance(content, dict) else None
        )
        if url:
            return f"fetched {url}"
        return "(result)"


# -----------------------------------------------------------------------------
# Convenience helper for quick scripted usage
# -----------------------------------------------------------------------------

def chat_and_collect(
    agent: AWSDocsAgent,
    history: list[dict[str, Any]],
    user_message: str,
) -> tuple[str, list[AgentEvent]]:
    """
    Append `user_message` to history, run the agent, and return (final_text, events).
    Mutates `history` in place to include the new user + assistant turns.
    """
    history.append({"role": "user", "content": user_message})
    events: list[AgentEvent] = []
    text_chunks: list[str] = []
    for event in agent.chat(history):
        events.append(event)
        if isinstance(event, TextEvent):
            text_chunks.append(event.text)
    return "\n".join(text_chunks).strip(), events


if __name__ == "__main__":
    # Smoke test: requires ANTHROPIC_API_KEY in env
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Set ANTHROPIC_API_KEY to run the smoke test.")
        raise SystemExit(0)
    agent = AWSDocsAgent()
    history: list[dict[str, Any]] = []
    answer, events = chat_and_collect(
        agent, history, "What is the maximum object size in S3? Cite the docs."
    )
    print("--- events ---")
    for e in events:
        print(e)
    print("\n--- answer ---")
    print(answer)
