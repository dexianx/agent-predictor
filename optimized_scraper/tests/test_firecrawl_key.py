from optimized_scraper.firecrawl_fallback import get_firecrawl_api_key


def test_firecrawl_key_loads_from_environment(monkeypatch):
    monkeypatch.setenv("FIRECRAWL_API_KEY", "test-key")
    assert get_firecrawl_api_key() == "test-key"


def test_firecrawl_key_is_not_hardcoded(monkeypatch):
    monkeypatch.delenv("FIRECRAWL_API_KEY", raising=False)
    assert get_firecrawl_api_key() is None
