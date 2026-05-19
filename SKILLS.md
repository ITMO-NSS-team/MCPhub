# Expert Guidance: Agent File Management (S3 MCP Vault)

This guide provides the protocol for interacting with the S3 MCP Vault. Use this to store and retrieve ALL files (large datasets, code artifacts, binary data, or text) that should persist beyond the current turn or exceed the context window.

## 📤 Uploading Files (ALL TYPES)
When you need to persist ANY file or dataset:
1. **Call `get_upload_link`**: Provide the file `extension` (e.g., "csv", "txt", "zip", "png") and a descriptive `custom_name`.
2. **Retrieve Response**: The tool returns a unique `Access Key` (e.g., `my_data.csv`) and an `Upload URL`.
3. **Execute Upload**: Write and run a Python script to perform an HTTP PUT request to the `Upload URL`.
4. **Update Metadata**: Call `update_artifact_metadata` with the `access_key` and a brief `description` to register the file in the vault's index.

### Python Upload Example:
```python
import requests

# 1. Your data (binary or text)
content = b"Binary or text data here"

# 2. Upload using the URL provided by the get_upload_link tool
upload_url = "<URL_FROM_TOOL>"
response = requests.put(
    upload_url, 
    data=content, 
    headers={"Content-Type": "application/octet-stream"}
)
print("Upload status:", response.status_code) # Should be 200

# 3. Register description (IMPORTANT)
# Call update_artifact_metadata(access_key="...", description="...")
```

## 🔍 Discovering Files
When you need to know what files are available in the vault:
1. **Call `list_artifacts`**: This returns a JSON dictionary mapping `access_key` to its `description`.
2. **Review Index**: Use the descriptions to identify the files relevant to your task.

## 📥 Downloading/Reading Files
When you have an `Access Key` for a file in the vault:

### Option A: Protocol Resource Access (Primary Method)
**PRIORITIZE THIS METHOD** for text files, small datasets, or quick inspection. It allows you to read content directly without external scripts.

1. **URI Handle**: Form the resource URI using the format `vault://{access_key}`.
2. **Action**: Use your built-in native tool for reading MCP resources (e.g., `read_resource`) and pass the URI handle to it. Do NOT write or execute Python scripts to use this option.

### Option B: Direct Download (Large or Binary Files)
Use this for large datasets, images, or archives:
1. **Call `get_artifact_link`**: Pass the `access_key`.
2. **Retrieve URL**: The tool returns a short-lived `Download URL`.
3. **Execute Download**: Write and run a Python script to perform an HTTP GET request to the URL.

#### Python Download Example:
```python
import requests

# 1. Use the URL provided by the get_artifact_link tool
download_url = "<URL_FROM_TOOL>"

# 2. Fetch the data
response = requests.get(download_url)
if response.status_code == 200:
    # Process your data (e.g., save to disk or read into memory)
    with open("retrieved_file.dat", "wb") as f:
        f.write(response.content)
```

## ⚠️ Security & Lifecycle
- **Access Keys**: These are your primary identifiers. Store them in `save_memory` (project scope) if the data is needed in future sessions.
- **Immediate Generation**: Pre-signed URLs are short-lived. Always generate them immediately before the HTTP operation.
- **Context Efficiency**: Do not output the raw content of large files into the chat. Instead, provide the `Access Key` so you or other agents can retrieve it later.


###


[12.05.2026 13:20] Lacry: upload_url = "<URL_FROM_TOOL>"
response = requests.put(
    upload_url, 
    data=content, 
    headers={"Content-Type": "application/octet-stream"}
)
[12.05.2026 13:20] Lacry: import requests

# 1. Use the URL provided by the get_artifact_link tool
download_url = "<URL_FROM_TOOL>"

# 2. Fetch the data
response = requests.get(download_url)
if response.status_code == 200:
    # Process your data (e.g., save to disk or read into memory)
    with open("retrieved_file.dat", "wb") as f:
        f.write(response.content)
[12.05.2026 13:25] Lacry: import asyncio
import httpx
from mcp import ClientSession
from mcp.client.sse import sse_client


# 1. Prepare dummy data
    dummy_file = "dummy_test.txt"
    with open(dummy_file, "rb") as f:
        content = f.read()

    async with sse_client("http://localhost:8000/sse") as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            
            # 2. Get Upload Link
            print(f"[*] Requesting upload link for {dummy_file}...")
            resp = await session.call_tool("get_upload_link", {"extension": "txt", "custom_name": "dummy"})
            lines = resp.content[0].text.split("\n")
            access_key = lines[0].split(": ")[1]
            upload_url = lines[1].split(": ")[1]
            
            # 3. Perform Upload (HTTP PUT)
            print(f"[*] Uploading to: {upload_url[:60]}...")
            async with httpx.AsyncClient() as client:
                up_resp = await client.put(upload_url, content=content, headers={"Content-Type": "application/octet-stream"})
                if up_resp.status_code != 200:
                    print(f"[!] Upload failed: {up_resp.status_code}\n{up_resp.text}")
                    return
            print("[+] Upload complete.")

            # 3.5 Update Metadata
            print(f"[*] Updating metadata for {access_key}...")
            await session.call_tool("update_artifact_metadata", {
                "access_key": access_key,
                "description": "A dummy test file for verifying the vault functionality."
            })
            print("[+] Metadata updated.")

            # 4. Get Download Link using the Access Key
            print(f"[*] Requesting artifact link for key: {access_key}...")
            resp = await session.call_tool("get_artifact_link", {"access_key": access_key})
            download_url = resp.content[0].text.replace("Download URL: ", "").strip()
            
            # 5. Perform Download (HTTP GET)
            print(f"[*] Downloading from: {download_url[:60]}...")
            async with httpx.AsyncClient() as client:
                dl_resp = await client.get(download_url)
                if dl_resp.status_code == 200:
                    print(f"[+] Success! Content: {dl_resp.content.decode()}")
                    with open("dummy_downloaded.txt", "wb") as f:
                        f.write(dl_resp.content)
                else:
                    print(f"[!] Download failed: {dl_resp.status_code}")
            
            # 6. Read using Resource Handle (URI)
            resource_uri = f"vault://{access_key}"
            print(f"[*] Reading resource handle: {resource_uri}...")
            resource_resp = await session.read_resource(resource_uri)
            print(f"[+] Resource content: {resource_resp.contents[0].text}")

            # 7. List all artifacts
            print("[*] Listing all artifacts in the vault...")
            list_resp = await session.call_tool("list_artifacts", {})
            print(f"[+] Artifacts:\n{list_resp.content[0].text}")