#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
source /root/miniconda3/bin/activate Mol_gen_env
nohup /root/miniconda3/envs/MCP_env/bin/python main_mcp.py > mcp.txt 2>&1 &
nohup python main_api.py > api.txt
