# MCP HTTP identity reuses the chat backend boundary

## Context

The chat HTTP adapter accepts a bearer token, resolves it through
`BackendIdentityResolver`, and passes the resulting user ID into the agent
runtime. The MCP adapter previously checked only whether a process-local token
existed. It did not resolve a subject, which made user-scoped tools fail and
would be unsafe for a shared HTTP MCP service.

## Decision

For HTTP MCP requests, extract the request bearer token and resolve it through
the same `BackendIdentityResolver` used by chat. The resolver calls the
backend's authenticated `/v1/me` endpoint, which remains the authority for
token validity and active-user status.

The token and transport mode are held in request-local `contextvars` while the
Streamable HTTP handler processes the request. Missing or invalid HTTP bearer
tokens produce an unauthenticated tool context and never fall back to a
container-wide local session. Stdio MCP retains the local OAuth-session flow
for developer use.

## Consequences

MCP and chat share the same backend identity contract without duplicating JWT
validation or sharing backend signing secrets with the agent. Tool code always
receives an authenticated `subject` for an HTTP call, and concurrent requests
cannot inherit another request's identity.
