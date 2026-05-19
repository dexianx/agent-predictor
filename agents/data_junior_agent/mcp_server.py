from __future__ import annotations

import logging

from fastmcp import FastMCP

from .tools.firecrawl_tools import register_firecrawl_tools

mcp = FastMCP(
    name="Data Junior MCP",
    instructions="Firecrawl tools for internet search and page extraction.",
)

register_firecrawl_tools(mcp)


def main() -> None:
    logging.getLogger("fastmcp").setLevel(logging.WARNING)
    mcp.run(transport="stdio", show_banner=False)


if __name__ == "__main__":
    main()
