"""Call start_generative_model_training on the remote MCP server."""
import asyncio
import json

from fastmcp import Client

SERVER_URL = "http://10.32.2.2:8882/mcp/"

TRAIN_URL = (
    "http://172.17.0.2:9000/molecule-generative-mcp/train/Alzheimer.csv"
    "?X-Amz-Algorithm=AWS4-HMAC-SHA256"
    "&X-Amz-Credential=UVPSAYE1E6F09N3LLI8G%2F20260518%2Fus-east-1%2Fs3%2Faws4_request"
    "&X-Amz-Date=20260518T151840Z"
    "&X-Amz-Expires=604800"
    "&X-Amz-Security-Token=eyJhbGciOiJIUzUxMiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3NLZXkiOiJVVlBTQVlFMUU2RjA5TjNMTEk4RyIsImV4cCI6MTc3OTEyMTA1MSwicGFyZW50IjoiY2hlbWNvc2NpZW50aXN0LXVzZXIifQ.ZvmElzmqq3QiRy-hDGaJtPIJpjzbSK0LFX7IaYuVv84myzpthD74YM2ZYO_Vpu9c9Z0kyX4hQm29ybdbUxXc9w"
    "&X-Amz-SignedHeaders=host"
    "&versionId=null"
    "&X-Amz-Signature=d5f5e2f86ed4173a539307a0a8b8c078b140af4b13783d8a3d189c0333cf6e45"
)


async def main() -> None:
    async with Client(SERVER_URL) as client:
        tools = await client.list_tools()
        print("Available tools:", [t.name for t in tools])

        result = await client.call_tool(
            "start_generative_model_training",
            {
                "case_name": "TEST_LINKS",
                "train_data_url": TRAIN_URL,
                "feature_column": ["canonical_smiles"],
            },
        )
        print("=== Result ===")
        try:
            print(json.dumps(result.data, indent=2, ensure_ascii=False))
        except Exception:
            print(result)


if __name__ == "__main__":
    asyncio.run(main())
