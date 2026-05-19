"""Probe MCP server availability over both https and http."""
import asyncio
import json

from fastmcp import Client

URLS = [
    "https://10.64.4.246:8779/mcp/",
    "http://10.64.4.246:8779/mcp/",
]


async def probe(url: str) -> None:
    print(f"--- {url} ---")
    try:
        async with Client(url) as client:
            tools = await client.list_tools()
            print(f"OK, {len(tools)} tools: {[t.name for t in tools]}")
    except Exception as exc:
        print(f"FAIL: {type(exc).__name__}: {exc}")


async def main() -> None:
    for url in URLS:
        await probe(url)


if __name__ == "__main__":
    asyncio.run(main())
