"""Train AutoML regression model for TEST_LINK on Synthetic Accessibility."""
import asyncio
import json

from fastmcp import Client

SERVER_URL = "http://localhost:8779/mcp/"

TRAIN_URL = (
    "http://10.32.1.114:9000/molecule-generative-mcp/generated/TEST_LINK/"
    "9d7b467657094618b32f674082b83fc4.csv"
    "?X-Amz-Algorithm=AWS4-HMAC-SHA256"
    "&X-Amz-Credential=chemcoscientist-user%2F20260519%2Fus-east-1%2Fs3%2Faws4_request"
    "&X-Amz-Date=20260519T120900Z"
    "&X-Amz-Expires=3600"
    "&X-Amz-SignedHeaders=host"
    "&X-Amz-Signature=15108403c4b13de59bd29fd7ab43c9b020062190eaeed820ee1249fc62f121a6"
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
        print("=== train_ml result ===")
        try:
            print(json.dumps(result.data, indent=2, ensure_ascii=False))
        except Exception:
            print(result)


if __name__ == "__main__":
    asyncio.run(main())
