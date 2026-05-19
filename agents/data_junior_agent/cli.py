from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
from typing import Any

os.environ.setdefault("LITELLM_LOG", "ERROR")

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

from .agent import list_mcp_tools, run_agent_stream
from .config import get_settings

console = Console()


def _quiet_dependency_logs() -> None:
    for name in ("fastmcp", "LiteLLM", "litellm", "mcp", "agents.mcp.util"):
        logging.getLogger(name).setLevel(logging.WARNING)


def cmd_config(_: argparse.Namespace) -> None:
    console.print_json(json.dumps(get_settings().redacted_dict(), indent=2))


def cmd_tools(_: argparse.Namespace) -> None:
    tools = asyncio.run(list_mcp_tools())
    table = Table(title="Data Junior MCP Tools")
    table.add_column("Tool", style="cyan")
    table.add_column("Description")
    for tool in tools:
        table.add_row(tool["name"], tool["description"])
    console.print(table)


def cmd_ask(args: argparse.Namespace) -> None:
    asyncio.run(_ask(args.prompt))


def cmd_chat(_: argparse.Namespace) -> None:
    console.print(Panel("Data Junior chat. Type 'exit' or 'quit' to leave.", title="Data Junior"))
    while True:
        prompt = Prompt.ask("[bold cyan]You[/bold cyan]").strip()
        if prompt.lower() in {"exit", "quit"}:
            break
        if not prompt:
            continue
        asyncio.run(_ask(prompt))


async def _ask(prompt: str) -> None:
    answer_parts: list[str] = []
    reasoning_parts: list[str] = []

    def render() -> str:
        answer = "".join(answer_parts)
        reasoning = "".join(reasoning_parts).strip()
        if reasoning:
            if len(reasoning) > 2400:
                reasoning = "..." + reasoning[-2400:]
            return f"[dim]Reasoning[/dim]\n{reasoning}\n\n[bold]Answer[/bold]\n{answer}"
        return answer

    with Live("", console=console, refresh_per_second=12) as live:
        try:
            async for event in run_agent_stream(prompt):
                if event.kind == "delta":
                    answer_parts.append(event.message)
                    live.update(render())
                elif event.kind == "reasoning_delta":
                    reasoning_parts.append(event.message)
                    live.update(render())
                elif event.kind == "final":
                    if not answer_parts:
                        answer_parts.append(event.message)
                    live.update(render())
                else:
                    console.print(f"[dim]{_label(event.kind)}[/dim] {event.message}")
        except Exception as exc:
            live.update("")
            console.print(Panel(_friendly_error(exc), title="Agent Error", border_style="red"))
    console.print()


def _friendly_error(exc: Exception) -> str:
    message = str(exc)
    if "Request too large" in message or "tokens per minute" in message or "TPM" in message:
        return (
            "The model provider rejected the request because the conversation became too large. "
            "I now compact Firecrawl tool output before it reaches the model; rerun the command. "
            "If it still happens, lower AGENT_TOOL_MAX_CHARS, for example:\n\n"
            "export AGENT_TOOL_MAX_CHARS=1200"
        )
    if "MaxTurnsExceeded" in exc.__class__.__name__ or "Max turns" in message:
        return (
            "The agent used its maximum number of reasoning/tool turns before producing a final answer. "
            "This usually happens when a tool is slow or the model keeps retrying sources. Rerun the "
            "query, ask for fewer sources, or raise AGENT_MAX_TURNS."
        )
    if "rate_limit" in message.lower() or "RateLimitError" in message:
        return f"The model provider rate-limited the request. Try again shortly.\n\n{message[:500]}"
    if "Cannot connect to host" in message or "nodename nor servname" in message:
        return (
            "The model provider could not be reached. This looks like a network or DNS issue, "
            "not a prompt/tool problem. Check your connection and rerun the command."
        )
    if "FIRECRAWL_API_KEY" in message:
        return (
            "Firecrawl is not configured for the tool process. Export FIRECRAWL_API_KEY in "
            "the same shell where you run the CLI, or put it in agents/.env."
        )
    return f"{exc.__class__.__name__}: {message[:1000]}"


def _label(kind: str) -> str:
    labels: dict[str, str] = {
        "summary": "reasoning",
        "tool_call": "tool",
        "tool_result": "tool",
    }
    return labels.get(kind, kind)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Chat with the Data Junior agent.")
    sub = parser.add_subparsers(dest="command", required=True)

    ask = sub.add_parser("ask", help="Ask one question and exit.")
    ask.add_argument("prompt")
    ask.set_defaults(func=cmd_ask)

    chat = sub.add_parser("chat", help="Start an interactive chat.")
    chat.set_defaults(func=cmd_chat)

    tools = sub.add_parser("tools", help="List MCP tools exposed to the agent.")
    tools.set_defaults(func=cmd_tools)

    config = sub.add_parser("config", help="Show resolved configuration with secrets redacted.")
    config.set_defaults(func=cmd_config)
    return parser


def main(argv: list[str] | None = None) -> None:
    _quiet_dependency_logs()
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
