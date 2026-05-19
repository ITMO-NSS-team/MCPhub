"""Check TEST_LINK ML training job status."""
import asyncio
import json

from fastmcp import Client

SERVER_URL = "http://localhost:8779/mcp/"
JOB_ID = "b200cc6da3f84f7cafeb5bd83484062e"


async def main() -> None:
    async with Client(SERVER_URL) as client:
        job = await client.call_tool("train_ml_job_status", {"job_id": JOB_ID})
        print("=== job status ===")
        print(json.dumps(job.data, indent=2, ensure_ascii=False))

        state = await client.call_tool("check_state", {})
        print("\n=== state[TEST_LINK] ===")
        case = state.data.get("state", {}).get("TEST_LINK")
        print(json.dumps(case, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main())
