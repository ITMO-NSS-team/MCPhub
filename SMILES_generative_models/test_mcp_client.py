from __future__ import annotations

import argparse
import asyncio
import json
import os
from typing import Any, Dict

from fastmcp.client import Client


def _as_jsonable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "dict"):
        return value.dict()
    if isinstance(value, (dict, list, str, int, float, bool)) or value is None:
        return value
    return str(value)


def _print_json(value: Any) -> None:
    print(json.dumps(_as_jsonable(value), ensure_ascii=False, indent=2))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test FastMCP server as an agent client (list tools, call tools)."
    )
    parser.add_argument(
        "--url",
        default=os.getenv("MCP_URL", "http://10.32.2.2:8883/mcp/"),
        help="MCP server URL (default: MCP_URL or http://10.32.2.2:8883/mcp/)",
    )
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="Only list tools and exit.",
    )
    parser.add_argument(
        "--call",
        help="Tool name to call after listing tools (e.g. get_state_from_server).",
    )
    parser.add_argument(
        "--args",
        default="{}",
        help="JSON object with tool arguments (e.g. '{\"url\":\"pred\"}').",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="Client timeout in seconds (optional).",
    )
    return parser.parse_args()


async def _run() -> None:
    args = _parse_args()
    try:
        tool_args: Dict[str, Any] = json.loads(args.args)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid JSON for --args: {exc}") from exc

    client = Client(args.url, timeout=args.timeout)
    async with client:
        tools = await client.list_tools()
        print("Available tools:")
        for tool in tools:
            name = getattr(tool, "name", "<unknown>")
            description = getattr(tool, "description", "")
            print(f"- {name}: {description}")

        if args.list_only:
            return

        if args.call:
            print(f"\nCalling tool: {args.call}")
            result = await client.call_tool(args.call, tool_args)
            _print_json(result)


def main() -> None:
    asyncio.run(_run())


if __name__ == "__main__":
    main()
