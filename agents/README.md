# Data Junior Agent

Data Junior is a local CLI agent for web research using Firecrawl-backed MCP
tools.

The implementation uses:

- OpenAI Agents SDK for the agent loop
- LiteLLM for provider routing
- FastMCP for local tools
- Pydantic for settings and typed tool data

## Setup

Use Python 3.11 if it is available:

```bash
cd /Users/fadex/Downloads/coding-apps/data-scraping/agents
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Keys

Export keys in your shell or copy `.env.example` to `agents/.env`.
The real `.env` file is ignored by git.

```bash
export GROQ_API_KEY="your_groq_key"
export FIRECRAWL_API_KEY="your_firecrawl_key"
export AGENT_MODEL="groq/openai/gpt-oss-120b"
export AGENT_MCP_TIMEOUT_SECONDS=30
export AGENT_TOOL_MAX_CHARS=2500
export AGENT_MAX_SCRAPES=2
export AGENT_MAX_SEARCHES=3
```

`AGENT_TOOL_MAX_CHARS` caps Firecrawl tool payloads before they are sent back
to the model. Lower it, for example to `1200`, if Groq reports TPM or request
size errors on a large research task.

`AGENT_MAX_SCRAPES` and `AGENT_MAX_SEARCHES` cap tool calls per agent run so a
slow site cannot make the agent loop until it hits `AGENT_MAX_TURNS`.

Gemini can be used later by setting `GEMINI_API_KEY` and a Gemini LiteLLM model
string such as `AGENT_MODEL="gemini/gemini-3.1-flash"`.

## Commands

```bash
python -m data_junior_agent.cli config
python -m data_junior_agent.cli tools
python -m data_junior_agent.cli ask "Search the web for Nigerian ecommerce datasets and summarize useful sources."
python -m data_junior_agent.cli chat
```

The CLI shows reasoning summaries and tool activity. It does not print hidden
chain-of-thought or API keys.

## Tests

Run the local tests:

```bash
cd /Users/fadex/Downloads/coding-apps/data-scraping
python -m pytest optimized_scraper/tests agents/tests -q
```

Run live provider checks only when you intentionally export keys:

```bash
AGENT_LIVE_TESTS=1 python -m pytest agents/tests -q -m live
```
