"""Generate molecules via remote MCP server."""
import asyncio
import json

from fastmcp import Client

SERVER_URL = "http://10.32.2.2:8882/mcp/"


async def main() -> None:
    async with Client(SERVER_URL) as client:
        result = await client.call_tool(
            "generate_mols",
            {"num": 400},
        )
        print("=== generate_mols result ===")
        try:
            print(json.dumps(result.data, indent=2, ensure_ascii=False))
        except Exception:
            print(result)


if __name__ == "__main__":
    asyncio.run(main())
