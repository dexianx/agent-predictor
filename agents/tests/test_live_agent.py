import os

import pytest

from data_junior_agent.agent import run_agent_stream
from data_junior_agent.config import get_settings
from data_junior_agent.tools.firecrawl_tools import firecrawl_scrape_url


pytestmark = pytest.mark.live


@pytest.mark.asyncio
async def test_live_agent_answers_with_groq():
    if os.environ.get("AGENT_LIVE_TESTS") != "1":
        pytest.skip("Set AGENT_LIVE_TESTS=1 to run live provider tests.")
    if not os.environ.get("GROQ_API_KEY"):
        pytest.skip("GROQ_API_KEY is required for the live Groq test.")

    final = ""
    try:
        async for event in run_agent_stream("Say 'Data Junior live check passed' and nothing else."):
            if event.kind == "final":
                final = event.message
    except Exception as exc:
        if _is_transient_network_error(exc):
            pytest.skip("Transient provider/network error during live Groq test.")
        raise

    assert "Data Junior live check passed" in final


def test_live_firecrawl_scrapes_example_page():
    if os.environ.get("AGENT_LIVE_TESTS") != "1":
        pytest.skip("Set AGENT_LIVE_TESTS=1 to run live provider tests.")
    if not os.environ.get("FIRECRAWL_API_KEY"):
        pytest.skip("FIRECRAWL_API_KEY is required for the live Firecrawl test.")

    get_settings.cache_clear()
    try:
        result = firecrawl_scrape_url("https://example.com", formats=["markdown"])
    except Exception as exc:
        if _is_transient_network_error(exc):
            pytest.skip("Transient network error during live Firecrawl test.")
        raise

    assert result
    assert "example" in str(result).lower()


def _is_transient_network_error(exc: Exception) -> bool:
    text = str(exc)
    return any(
        marker in text
        for marker in (
            "Cannot connect to host",
            "nodename nor servname",
            "Temporary failure in name resolution",
            "Name or service not known",
        )
    )
