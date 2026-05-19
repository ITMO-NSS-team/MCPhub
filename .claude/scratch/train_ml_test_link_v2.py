"""Train AutoML regression model for TEST_LINK on fresh generated CSV."""
import asyncio
import json

from fastmcp import Client

SERVER_URL = "http://localhost:8779/mcp/"

TRAIN_URL = (
    "http://10.32.1.114:9000/molecule-generative-mcp/generated/TEST_LINK/"
    "436b809af4794cccb165e7e6a6d1d12b.csv"
    "?X-Amz-Algorithm=AWS4-HMAC-SHA256"
    "&X-Amz-Credential=chemcoscientist-user%2F20260519%2Fus-east-1%2Fs3%2Faws4_request"
    "&X-Amz-Date=20260519T160824Z"
    "&X-Amz-Expires=3600"
    "&X-Amz-SignedHeaders=host"
    "&X-Amz-Signature=05407305c3b1db5ca0e28ee0275c07913469528a925a732a28e08180631b4f16"
)


async def main() -> None:
    async with Client(SERVER_URL) as client:
        result = await client.call_tool(
            "train_ml",
            {
                "case": "TEST_LINK",
                "train_data_url": TRAIN_URL,
                "feature_column": ["Smiles"],
                "target_column": ["Synthetic Accessibility"],
                "regression_props": ["Synthetic Accessibility"],
            },
        )
        print(json.dumps(result.data, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main())
