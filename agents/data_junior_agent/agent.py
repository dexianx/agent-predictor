from __future__ import annotations

import os
import sys
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

from .config import AgentSettings, get_settings

TOOL_NAMES = {
    "firecrawl_scrape_url",
    "firecrawl_search",
}

INSTRUCTIONS = """
You are Data Junior, an agent for researching web sources with Firecrawl-backed MCP tools.

Use the MCP tools when the user asks for current web information or page extraction.
Think privately. When the configured model exposes reasoning text, stream it in the
separate reasoning lane. When helpful, cite tool results in plain language and give
clear next steps.

For broad web research, start with firecrawl_search and summarize from those results
unless a page scrape is clearly necessary. Scrape at most two URLs for a single answer.
If a scrape times out or returns an error, stop retrying that source and continue with
the search snippets and any successful scrape results.

Do not invent exact row counts, file sizes, schemas, dates, or license terms unless
they appear in tool results. If a source only appears in search results, summarize it
as a candidate source and say that details need verification.
""".strip()


@dataclass(slots=True)
class AgentEvent:
    kind: str
    message: str
    data: Any | None = None


def _configure_litellm_env(settings: AgentSettings) -> None:
    api_key = settings.require_model_api_key()
    model = settings.agent_model.lower()
    if model.startswith("groq/"):
        os.environ.setdefault("GROQ_API_KEY", api_key)
    elif model.startswith("gemini/"):
        os.environ.setdefault("GEMINI_API_KEY", api_key)
    elif model.startswith("xai/") or model.startswith("grok/"):
        os.environ.setdefault("XAI_API_KEY", api_key)
    elif model.startswith("openai/") or "/" not in model:
        os.environ.setdefault("OPENAI_API_KEY", api_key)


def build_model(settings: AgentSettings | None = None) -> Any:
    settings = settings or get_settings()
    _configure_litellm_env(settings)
    from agents.extensions.models.litellm_model import LitellmModel

    return LitellmModel(model=settings.agent_model, api_key=settings.require_model_api_key())


def build_agent(mcp_server: Any, settings: AgentSettings | None = None) -> Any:
    settings = settings or get_settings()
    from agents import Agent

    return Agent(
        name="Data Junior",
        instructions=INSTRUCTIONS,
        model=build_model(settings),
        mcp_servers=[mcp_server],
    )


def build_mcp_server() -> Any:
    from agents.mcp import MCPServerStdio, create_static_tool_filter

    child_env = os.environ.copy()
    settings = get_settings()
    firecrawl_key = settings.firecrawl_key_value()
    if firecrawl_key:
        child_env["FIRECRAWL_API_KEY"] = firecrawl_key

    return MCPServerStdio(
        name="Data Junior MCP",
        params={
            "command": sys.executable,
            "args": ["-m", "data_junior_agent.mcp_server"],
            "cwd": str(__import__("pathlib").Path(__file__).resolve().parents[1]),
            "env": child_env,
        },
        cache_tools_list=True,
        client_session_timeout_seconds=get_settings().agent_mcp_timeout_seconds,
        tool_filter=create_static_tool_filter(allowed_tool_names=sorted(TOOL_NAMES)),
        failure_error_function=_format_mcp_tool_error,
    )


async def run_agent_stream(
    prompt: str,
    settings: AgentSettings | None = None,
) -> AsyncIterator[AgentEvent]:
    settings = settings or get_settings()
    if not settings.agent_trace:
        from agents import set_tracing_disabled

        set_tracing_disabled(True)

    async with build_mcp_server() as mcp_server:
        agent = build_agent(mcp_server, settings)
        from agents import Runner

        result = Runner.run_streamed(agent, input=prompt, max_turns=settings.agent_max_turns)
        yield AgentEvent("summary", "Planning request and selecting tools if needed.")
        async for event in result.stream_events():
            summary = _summarize_stream_event(event)
            if summary is not None:
                yield summary
        if getattr(result, "final_output", None):
            yield AgentEvent("final", str(result.final_output))


async def list_mcp_tools() -> list[dict[str, Any]]:
    async with build_mcp_server() as server:
        tools = await server.list_tools()
    results = []
    for tool in tools:
        results.append(
            {
                "name": getattr(tool, "name", ""),
                "description": getattr(tool, "description", "") or "",
            }
        )
    return results


def _summarize_stream_event(event: Any) -> AgentEvent | None:
    event_type = getattr(event, "type", "")
    if event_type == "raw_response_event":
        data = getattr(event, "data", None)
        data_type = getattr(data, "type", "")
        data_class = data.__class__.__name__ if data is not None else ""
        if "reasoning" in data_type or "Reasoning" in data_class:
            delta = getattr(data, "delta", None)
            if delta:
                return AgentEvent("reasoning_delta", str(delta))
            return None
        if data_type != "response.output_text.delta" and data_class != "ResponseTextDeltaEvent":
            return None
        delta = getattr(data, "delta", None)
        if delta:
            return AgentEvent("delta", str(delta))
        return None
    if event_type == "run_item_stream_event":
        name = getattr(event, "name", "")
        item = getattr(event, "item", None)
        if name in {"tool_called", "tool_call_item"}:
            return AgentEvent("tool_call", _tool_name(item) or "Calling tool.")
        if name in {"tool_output", "tool_call_output_item"}:
            return AgentEvent("tool_result", _short_tool_output(item))
        if name == "message_output_created":
            return None
        if "reasoning" in name:
            return AgentEvent("summary", name.replace("_", " ").capitalize())
        if name:
            return AgentEvent("summary", name.replace("_", " ").capitalize())
    if event_type == "agent_updated_stream_event":
        agent = getattr(event, "new_agent", None)
        name = getattr(agent, "name", None)
        if name:
            return AgentEvent("summary", f"Using agent: {name}.")
    return None


def _format_mcp_tool_error(_context: Any, exc: Exception) -> str:
    message = str(exc)
    if "Timed out while waiting for response" in message:
        return (
            "Firecrawl timed out while loading that source. Do not retry the same page; "
            "continue from existing search results and successful scrapes."
        )
    if "budget exhausted" in message.lower():
        return message
    if "FIRECRAWL_API_KEY" in message:
        return (
            "Firecrawl is not configured. Ask the user to export FIRECRAWL_API_KEY in the "
            "same shell or add it to agents/.env."
        )
    return f"Tool failed: {message[:500]}"


def _tool_name(item: Any) -> str | None:
    raw_item = getattr(item, "raw_item", item)
    for attr in ("name", "tool_name"):
        value = getattr(raw_item, attr, None) or getattr(item, attr, None)
        if value:
            return f"Calling tool: {value}"
    return None


def _short_tool_output(item: Any) -> str:
    output = getattr(item, "output", None)
    if output is None:
        output = getattr(getattr(item, "raw_item", item), "output", "")
    text = str(output)
    if len(text) > 260:
        text = text[:257] + "..."
    return f"Tool returned: {text}"
