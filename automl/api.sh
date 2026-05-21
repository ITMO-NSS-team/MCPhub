#!/bin/bash
# Start the AutoML MCP server in the container.
# Output (stdout + stderr) is captured to `mcp.txt` next to the script so it
# can be read back by the `get_mcp_logs` MCP tool. Runs in the foreground —
# `nohup` only insulates from SIGHUP; without `&` the script blocks here so
# the container stays alive on the python process.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
nohup python automl_mcp.py > mcp.txt 2>&1
