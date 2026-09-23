---
id: VICTUS-CONTRACT-PUBLIC-DEMO-V1
contract_id: victus.contract.agent.public-demo.v1
status: current
version: v1
owner: victus-agent-runtime
---

# Public read-only demo

`POST /demo/chat` is for the Victus WebApp backend only. Browsers must never call it directly.

## Authorization

The backend signs a one-time ES256 JWT with its private key. The agent receives only an ES256 public
JWKS in `VICTUS_DEMO_JWT_JWKS_JSON`; every key must contain a rotation `kid`.

Required claims are `iss=victus-webapp`, `sub=demo:david`, `aud=victus-agent`, `demo=true`, both
`demo:chat` and `demo:read` scopes, `profile_version=david-v1`, a stable opaque `sid`, `iat`, `exp`,
and a unique `jti`.
The lifetime may not exceed 30 seconds. Invalid tokens return `401`, policy claims return `403`, and an unsupported
fixture version returns `422`.

## Request and response

```json
{"conversation_id":"ephemeral-browser-session-id","request_id":"uuid","message":"Can David change lunch?","language":"en"}
```

```json
{"message":"…","profile_version":"david-v1","read_only":true}
```

`conversation_id` is browser metadata only and is never used for authorization or state lookup. The
WebApp creates an opaque `sid` when the page session starts, includes it in every short-lived signed
JWT, and creates a new `sid` when the page reloads. The agent uses only that signed value to isolate
the ephemeral LangGraph thread.

## Data and tool policy

The demo runs the standard LangGraph topology with its normal safety and tool-decision behavior.
Only `src/demo/fixtures/david-v1.json` is available as profile context. Its tool policy is explicit:

- `event_capture` is enabled but writes only to a session-local in-memory event sink.
- `evidence_retrieval` is enabled as a read-only Victus RAG request.
- Database-backed tools, profile lookup, account/session changes, files, webhooks, and all unspecified
  tools are denied.

Checkpoints and explicit memory requests live only in per-`sid` in-memory resources with a bounded
TTL. Demo telemetry uses the same input, output, LLM, and tool visibility as production, tagged with
`victus.demo=true`, `victus.execution_mode=demo`, and the fixture version; Phoenix must therefore be
treated as internal trusted infrastructure. `read_only: true` means the fixture and every durable
system remain unchanged; it does not prevent temporary in-session state.
