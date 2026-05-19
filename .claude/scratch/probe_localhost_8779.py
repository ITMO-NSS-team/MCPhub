"""Probe MCP server on localhost:8779 — list tools."""
import asyncio
import json

from fastmcp import Client

SERVER_URL = "http://localhost:8779/mcp/"


async def main() -> None:
    async with Client(SERVER_URL) as client:
        tools = await client.list_tools()
        print(f"Connected. {len(tools)} tools available:")
        for t in tools:
            desc = (t.description or "").strip().splitlines()[0] if t.description else ""
            print(f"  - {t.name}: {desc[:120]}")


if __name__ == "__main__":
    asyncio.run(main())
