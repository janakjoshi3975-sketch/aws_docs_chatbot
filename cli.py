"""
Terminal CLI for the AWS Documentation Agent.

Run:
    python cli.py
    python cli.py "What is the max object size in S3?"   # one-shot

Type /reset to clear history, /quit to exit.
"""

from __future__ import annotations

import argparse
import os
import sys

from agent import (
    AWSDocsAgent,
    ClientToolEvent,
    FinalEvent,
    ServerToolEvent,
    ServerToolResultEvent,
    TextEvent,
)


# ANSI colors (no extra deps)
DIM = "\033[2m"
BOLD = "\033[1m"
ORANGE = "\033[38;5;208m"
BLUE = "\033[38;5;39m"
RESET = "\033[0m"


def print_event(event) -> None:
    if isinstance(event, TextEvent):
        sys.stdout.write(event.text)
        sys.stdout.flush()
    elif isinstance(event, ServerToolEvent):
        q = event.input.get("query") or event.input.get("url") or str(event.input)
        icon = "🔍" if event.name == "web_search" else "📄"
        print(f"\n{DIM}{icon} {event.name}: {q}{RESET}")
    elif isinstance(event, ServerToolResultEvent):
        print(f"{DIM}   ↳ {event.summary}{RESET}")
    elif isinstance(event, ClientToolEvent):
        print(f"{DIM}🛠️  {event.name}({event.input}) → {event.result}{RESET}")
    elif isinstance(event, FinalEvent):
        usage = event.usage or {}
        if usage:
            print(
                f"\n{DIM}[stop={event.stop_reason} "
                f"in={usage.get('input_tokens', '?')} "
                f"out={usage.get('output_tokens', '?')}]{RESET}"
            )


def run_turn(agent: AWSDocsAgent, history: list, user_message: str) -> None:
    history.append({"role": "user", "content": user_message})
    print(f"\n{ORANGE}{BOLD}assistant ▸{RESET} ", end="")
    for event in agent.chat(history):
        print_event(event)
    print()  # newline after the answer


def main() -> None:
    parser = argparse.ArgumentParser(description="AWS Docs Agent CLI")
    parser.add_argument(
        "question",
        nargs="*",
        help="Optional one-shot question. If omitted, starts an interactive session.",
    )
    parser.add_argument("--model", default=None)
    parser.add_argument("--notes", default=None)
    args = parser.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY is not set.", file=sys.stderr)
        sys.exit(1)

    kwargs = {}
    if args.model:
        kwargs["model"] = args.model
    if args.notes:
        kwargs["notes_path"] = args.notes
    agent = AWSDocsAgent(**kwargs)
    history: list = []

    if args.question:
        run_turn(agent, history, " ".join(args.question))
        return

    print(f"{BLUE}{BOLD}AWS Docs Agent{RESET}  {DIM}(/reset, /quit){RESET}")
    while True:
        try:
            user_message = input(f"\n{BLUE}{BOLD}you ▸{RESET} ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user_message:
            continue
        if user_message in ("/quit", "/exit"):
            break
        if user_message == "/reset":
            history.clear()
            print(f"{DIM}(history cleared){RESET}")
            continue
        try:
            run_turn(agent, history, user_message)
        except Exception as exc:  # noqa: BLE001
            print(f"\n{DIM}error: {exc!r}{RESET}")


if __name__ == "__main__":
    main()
