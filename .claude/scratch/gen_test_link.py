"""Generate 10 molecules using TEST_LINK case."""
import asyncio
import json

from fastmcp import Client

SERVER_URL = "http://10.32.2.2:8882/mcp/"


async def main() -> None:
    async with Client(SERVER_URL) as client:
        result = await client.call_tool(
            "generate_mols",
            {"num": 10, "case": "TEST_LINK"},
        )
        print("=== generate_mols (TEST_LINK) ===")
        try:
            print(json.dumps(result.data, indent=2, ensure_ascii=False))
        except Exception:
            print(result)


if __name__ == "__main__":
    asyncio.run(main())
