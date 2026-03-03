import asyncio
from fastmcp import Client, FastMCP

# In-memory server (ideal for testing)
server = FastMCP("TestServer")
client = Client(server)



# Local Python script
client = Client("http://10.32.2.2:8883/mcp")

async def main():
    async with client:
        # Basic server interaction
        await client.ping()

        # List available operations
        tools = await client.list_tools()
        resources = await client.list_resources()
        prompts = await client.list_prompts()
        print(tools,resources,prompts)
        # Execute operations
        result = await client.call_tool("get_state_from_server", {})
        print(result)

asyncio.run(main())