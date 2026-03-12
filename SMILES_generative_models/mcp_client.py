import asyncio
from fastmcp import Client, FastMCP

# In-memory server (ideal for testing)
server = FastMCP("TestServer")
client = Client(server)



# Local Python script
client = Client("http://10.32.2.2:8884/mcp")

async def main():
    async with client:
        # Basic server interaction
        await client.ping()

        # List available operations
        tools = await client.list_tools()
        resources = await client.list_resources()
        prompts = await client.list_prompts()
        print("##################tools#################")
        print(tools)
        print("###################################")
        print(resources)
        print(prompts)
        # Execute operations
        #result = await client.call_tool("start_generative_model_training",{"case_name": "Test_mas_1", "feature_column":["_smiles"]})
        #result = await client.call_tool("list_s3_train_cases",)
        result = await client.call_tool("get_state_from_server",{"case":"Test_mas_1"})
        print(result)

asyncio.run(main())