"""Probe MCP at http://10.64.4.1:8779/mcp/."""
import asyncio
from fastmcp import Client

URL = "http://10.64.4.1:8779/mcp/"


async def main() -> None:
    try:
        async with Client(URL) as client:
            tools = await client.list_tools()
            print(f"OK, {len(tools)} tools: {[t.name for t in tools]}")
    except Exception as exc:
        print(f"FAIL: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    asyncio.run(main())
