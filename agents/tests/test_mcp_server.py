import pytest

from data_junior_agent.agent import build_mcp_server


@pytest.mark.asyncio
async def test_mcp_server_lists_dataset_tools():
    async with build_mcp_server() as server:
        tools = await server.list_tools()

    names = {tool.name for tool in tools}
    assert names == {"firecrawl_scrape_url", "firecrawl_search"}
