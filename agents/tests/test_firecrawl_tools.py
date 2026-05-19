from data_junior_agent.config import get_settings
from data_junior_agent.tools import firecrawl_tools


class FakeFirecrawlApp:
    def __init__(self, api_key):
        self.api_key = api_key

    def scrape_url(self, url, formats, only_main_content):
        return {
            "url": url,
            "formats": formats,
            "only_main_content": only_main_content,
            "markdown": "# Example",
        }

    def search(self, query, limit):
        return {"query": query, "data": [{"title": "Example", "url": "https://example.com"}][:limit]}


def test_firecrawl_scrape_url_uses_client(monkeypatch):
    monkeypatch.setenv("FIRECRAWL_API_KEY", "fc_test")
    get_settings.cache_clear()
    firecrawl_tools._reset_tool_budgets_for_tests()
    monkeypatch.setattr(firecrawl_tools, "_client", lambda: FakeFirecrawlApp("fc_test"))

    result = firecrawl_tools.firecrawl_scrape_url("https://example.com")

    assert result["url"] == "https://example.com"
    assert result["markdown"] == "# Example"


def test_firecrawl_search_uses_client(monkeypatch):
    monkeypatch.setenv("FIRECRAWL_API_KEY", "fc_test")
    get_settings.cache_clear()
    firecrawl_tools._reset_tool_budgets_for_tests()
    monkeypatch.setattr(firecrawl_tools, "_client", lambda: FakeFirecrawlApp("fc_test"))

    result = firecrawl_tools.firecrawl_search("nigerian ecommerce", limit=1)

    assert result["query"] == "nigerian ecommerce"
    assert result["data"][0]["url"] == "https://example.com"


def test_firecrawl_scrape_url_compacts_large_markdown(monkeypatch):
    monkeypatch.setenv("FIRECRAWL_API_KEY", "fc_test")
    monkeypatch.setenv("AGENT_TOOL_MAX_CHARS", "1200")
    get_settings.cache_clear()
    firecrawl_tools._reset_tool_budgets_for_tests()
    monkeypatch.setattr(firecrawl_tools, "_client", lambda: FakeFirecrawlApp("fc_test"))

    class LargeFakeFirecrawlApp(FakeFirecrawlApp):
        def scrape_url(self, url, formats, only_main_content):
            return {"url": url, "markdown": "x" * 5000}

    monkeypatch.setattr(firecrawl_tools, "_client", lambda: LargeFakeFirecrawlApp("fc_test"))
    result = firecrawl_tools.firecrawl_scrape_url("https://example.com")

    assert len(str(result)) < 1400
    assert "truncated" in str(result).lower()


def test_firecrawl_missing_key_returns_tool_error(monkeypatch):
    monkeypatch.delenv("FIRECRAWL_API_KEY", raising=False)
    get_settings.cache_clear()
    firecrawl_tools._reset_tool_budgets_for_tests()

    result = firecrawl_tools.firecrawl_search("nigerian ecommerce", limit=1)

    assert "error" in result
    assert "FIRECRAWL_API_KEY" in result["error"]


def test_firecrawl_scrape_budget_stops_repeated_scrapes(monkeypatch):
    monkeypatch.setenv("FIRECRAWL_API_KEY", "fc_test")
    monkeypatch.setenv("AGENT_MAX_SCRAPES", "1")
    get_settings.cache_clear()
    firecrawl_tools._reset_tool_budgets_for_tests()
    monkeypatch.setattr(firecrawl_tools, "_client", lambda: FakeFirecrawlApp("fc_test"))

    first = firecrawl_tools.firecrawl_scrape_url("https://example.com")
    second = firecrawl_tools.firecrawl_scrape_url("https://example.com/again")

    assert "markdown" in first
    assert second["error"] == "Scrape budget exhausted for this agent run."
