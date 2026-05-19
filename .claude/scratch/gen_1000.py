"""Generate 1000 molecules using TEST_LINK case."""
import asyncio
import json

from fastmcp import Client

SERVER_URL = "http://10.32.2.2:8882/mcp/"


async def main() -> None:
    async with Client(SERVER_URL) as client:
        result = await client.call_tool(
            "generate_mols",
            {"num": 1000, "case": "TEST_LINK"},
        )
        print(json.dumps(result.data, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main())
