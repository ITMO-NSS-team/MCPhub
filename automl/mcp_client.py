import asyncio
from fastmcp import Client, FastMCP

# In-memory server (ideal for testing)
server = FastMCP("TestServer")
client = Client(server)



# Local Python script
client = Client("http://localhost:8773/mcp")

async def main():
    async with client:
        # Basic server interaction
        await client.ping()

        # List available operations
        tools = await client.list_tools()
        resources = await client.list_resources()
        prompts = await client.list_prompts()
        #print(tools,resources,prompts)
        # Execute operations
        #result = await client.call_tool("check_state", {})
        #print(result)
        #result = await client.call_tool("predict_ml", {"case": "Alzheimer", "smiles_list": ["CCO"]})
        #result = await client.call_tool("get_s3_train_case_columns",{"case_name":"Test_mas_1"} )
        result = await client.call_tool("predict_ml", {"case": "Test_mas_1", "smiles_list": ["CCO"]})
        # result = await client.call_tool(
        #     "train_ml",
        #     {"case": "Test_mas_1", "feature_column": ["canonical_smiles"], "target_column": ["docking_score"],"regression_props":["docking_score"]},
        # )
        #result = await client.call_tool("train_ml_job_status",{"job_id":"cffdbcf4b82541d5bbfb9bac34ec1e8c",})
        #result = await client.call_tool("check_state")
        
        #result = await client.call_tool("check_state",)
        print(result)

asyncio.run(main())
