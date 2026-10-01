from __future__ import annotations

import os
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Any

from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Mount, Route

from adapters.http.auth import IdentityResolver
from adapters.mcp.auth import http_request_identity
from adapters.mcp.server import build_server
from bootstrap.storage import prepare_mcp_storage

DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8765
MCP_PATH = "/mcp"


def create_app(
    *,
    storage_preparer: Callable[[], Awaitable[None]] = prepare_mcp_storage,
    identity_resolver: IdentityResolver | None = None,
) -> Starlette:
    mcp_server = build_server(identity_resolver=identity_resolver)
    session_manager = StreamableHTTPSessionManager(
        app=mcp_server,
        json_response=True,
        stateless=True,
    )

    @asynccontextmanager
    async def lifespan(app: Starlette) -> AsyncIterator[None]:
        await storage_preparer()
        async with session_manager.run():
            yield

    async def health(request: Request) -> JSONResponse:
        return JSONResponse(
            {
                "status": "ok",
                "service": "victus-agent-mcp-http",
                "transport": "streamable_http",
            }
        )

    async def handle_mcp(scope: dict[str, Any], receive: Any, send: Any) -> None:
        with http_request_identity(_bearer_token(scope)):
            await session_manager.handle_request(scope, receive, send)

    return Starlette(
        debug=False,
        lifespan=lifespan,
        routes=[
            Route("/health", health, methods=["GET"]),
            Mount(MCP_PATH, app=handle_mcp),
        ],
    )


def _bearer_token(scope: dict[str, Any]) -> str | None:
    for raw_name, raw_value in scope.get("headers", []):
        if raw_name.lower() != b"authorization":
            continue
        try:
            scheme, token = raw_value.decode("latin-1").split(" ", 1)
        except ValueError:
            return None
        if scheme.lower() == "bearer" and token:
            return token
    return None


def main() -> None:
    import uvicorn

    host = os.getenv("VICTUS_MCP_HTTP_HOST", DEFAULT_HOST)
    port = int(os.getenv("VICTUS_MCP_HTTP_PORT", str(DEFAULT_PORT)))
    uvicorn.run(create_app(), host=host, port=port)


if __name__ == "__main__":
    main()
