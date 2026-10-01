from __future__ import annotations

from typing import Any

from adapters.http.auth import IdentityResolver
from adapters.mcp.auth import resolve_identity
from tools.contracts import ToolContext, ToolInvocation, ToolResult
from tools.runtime import ToolRuntime


async def invoke(
    runtime: ToolRuntime,
    name: str,
    arguments: dict[str, Any],
    *,
    identity_resolver: IdentityResolver | None = None,
) -> ToolResult:
    return await runtime.invoke_async(
        ToolInvocation(
            name=name,
            arguments=arguments,
            context=ToolContext(
                source="mcp", identity=await resolve_identity(resolver=identity_resolver)
            ),
        )
    )
