from __future__ import annotations

import asyncio
from typing import Any

from adapters.http.auth import IdentityResolver
from adapters.mcp.discovery import discover_tools
from adapters.mcp.invocation import invoke
from adapters.mcp.mapping import to_mcp_content
from bootstrap.runtime import build_runtime


def build_server(runtime=None, *, identity_resolver: IdentityResolver | None = None):
    import mcp.types as types
    from mcp.server.lowlevel import Server

    server = Server("victus-agent")
    runtime = runtime or build_runtime()

    @server.list_tools()
    async def handle_list_tools() -> list[types.Tool]:
        return discover_tools()

    @server.call_tool()
    async def handle_call_tool(name: str, arguments: dict[str, Any]) -> list[types.ContentBlock]:
        return to_mcp_content(
            await invoke(runtime, name, arguments, identity_resolver=identity_resolver)
        )

    return server


async def run_stdio() -> None:
    from mcp.server.lowlevel import NotificationOptions
    from mcp.server.models import InitializationOptions
    from mcp.server.stdio import stdio_server

    server = build_server()
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="victus-agent",
                server_version="0.1.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )


def main() -> None:
    asyncio.run(run_stdio())


if __name__ == "__main__":
    main()
