"""Check S3 train cases on automl MCP server."""
import asyncio
import json

from fastmcp import Client

SERVER_URL = "http://localhost:8779/mcp/"


async def main() -> None:
    async with Client(SERVER_URL) as client:
        result = await client.call_tool("list_s3_train_cases", {})
        print(json.dumps(result.data, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main())
