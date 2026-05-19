"""Inspect train_ml tool schema on localhost:8779."""
import asyncio
import json

from fastmcp import Client

SERVER_URL = "http://localhost:8779/mcp/"


async def main() -> None:
    async with Client(SERVER_URL) as client:
        tools = await client.list_tools()
        for t in tools:
            if t.name == "train_ml":
                print("=== train_ml input schema ===")
                print(json.dumps(t.inputSchema, indent=2, ensure_ascii=False))
                print()
                print("=== description ===")
                print(t.description)


if __name__ == "__main__":
    asyncio.run(main())
