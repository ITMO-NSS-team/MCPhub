"""Train generative model for case TEST_LINK using generated CSV URL."""
import asyncio
import json

from fastmcp import Client

SERVER_URL = "http://10.32.2.2:8882/mcp/"

TRAIN_URL = (
    "http://10.32.1.114:9000/molecule-generative-mcp/generated/gan_default/"
    "cc2256145b824e41988a2b3372e52ab6.csv"
    "?X-Amz-Algorithm=AWS4-HMAC-SHA256"
    "&X-Amz-Credential=chemcoscientist-user%2F20260519%2Fus-east-1%2Fs3%2Faws4_request"
    "&X-Amz-Date=20260519T104015Z"
    "&X-Amz-Expires=3600"
    "&X-Amz-SignedHeaders=host"
    "&X-Amz-Signature=cd3b75023ac8f81623b3ce092a3905c3adc733433ccc1a95f1649591e868f5b3"
)


async def main() -> None:
    async with Client(SERVER_URL) as client:
        result = await client.call_tool(
            "start_generative_model_training",
            {
                "case_name": "TEST_LINK",
                "train_data_url": TRAIN_URL,
                "feature_column": ["Smiles"],
            },
        )
        print("=== Result ===")
        try:
            print(json.dumps(result.data, indent=2, ensure_ascii=False))
        except Exception:
            print(result)


if __name__ == "__main__":
    asyncio.run(main())
