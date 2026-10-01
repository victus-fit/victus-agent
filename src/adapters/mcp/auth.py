from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from collections.abc import Iterator

from adapters.http.auth import BackendIdentityResolver, IdentityResolver
from tools.contracts import ToolIdentity
from victus_platform.identity.local_session import get_valid_access_token


_http_request_active: ContextVar[bool] = ContextVar("victus_mcp_http_request_active", default=False)
_request_bearer_token: ContextVar[str | None] = ContextVar(
    "victus_mcp_request_bearer_token", default=None
)


@contextmanager
def http_request_identity(token: str | None) -> Iterator[None]:
    """Bind an HTTP bearer to the current MCP request only."""
    active_token = _http_request_active.set(True)
    bearer_token = _request_bearer_token.set(token)
    try:
        yield
    finally:
        _request_bearer_token.reset(bearer_token)
        _http_request_active.reset(active_token)


async def resolve_identity(*, resolver: IdentityResolver | None = None) -> ToolIdentity:
    """Resolve the current caller without allowing HTTP cross-user fallback."""
    token = _request_bearer_token.get()
    if token is None and not _http_request_active.get():
        token = await get_valid_access_token()
    if not token:
        return ToolIdentity()

    try:
        subject = await (resolver or BackendIdentityResolver()).resolve(token)
    except RuntimeError:
        return ToolIdentity()
    return ToolIdentity(subject=subject, authenticated=bool(subject))
