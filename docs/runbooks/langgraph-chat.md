---
id: VICTUS-RUNBOOK-LANGGRAPH-CHAT
title: LangGraph Chat Operations
status: current
updated_at: 2026-07-18
owners:
  - victus-agent-runtime
---
# Setup

Create `.env` once, then configure `VICTUS_DOCKER_BACKEND_API_URL`,
`VICTUS_DOCKER_LITELLM_PROXY_API_BASE`, and the proxy key:

```bash
cp .env.example .env
docker compose up -d --build
docker compose ps
```

`agent` and `mcp` wait for PostgreSQL and apply pending application migrations during startup under
one shared database lock. The agent also prepares its LangGraph checkpoint and Store tables before
accepting traffic.

# Run

Check both HTTP services:

```bash
curl http://localhost:8766/health
curl http://localhost:8765/health
docker compose logs -f agent mcp
```

Stop all services while retaining PostgreSQL data with `docker compose down`. Use
`docker compose down -v` only when intentionally deleting local data.

Send a new turn with a backend-issued bearer token:

```bash
curl -X POST http://localhost:8766/chat \
  -H 'Authorization: Bearer <token>' \
  -H 'Content-Type: application/json' \
  -d '{"conversation_id":"demo-1","request_id":"turn-1","message":"hoy comi arroz"}'
```

For controlled debugging, enable the route before starting the service:

```bash
VICTUS_CHAT_DEBUG_ENABLED=true docker compose up -d --build agent
```

The webapp backend may then forward the same user JWT and inspect the bounded graph state:

```bash
curl -X POST http://localhost:8766/chat/debug \
  -H 'Authorization: Bearer <token>' \
  -H 'Content-Type: application/json' \
  -d '{"conversation_id":"debug-1","request_id":"turn-1","message":"preséntate brevemente"}'
```

Do not expose this endpoint directly to browsers or enable it broadly in production. The response
contains the authenticated user's own conversational and health context even though secret-bearing
field names are redacted and payload sizes are bounded.

When status is `needs_user_response`, resume the same conversation:

```bash
curl -X POST http://localhost:8766/chat \
  -H 'Authorization: Bearer <token>' \
  -H 'Content-Type: application/json' \
  -d '{"conversation_id":"demo-1","request_id":"turn-2","resume":{"value":{"accepted":true}}}'
```

# Recovery

- A request may be retried with the same `request_id`; tool idempotency keys are stable per loop.
- Restore PostgreSQL from backup to recover checkpoints, Store memory, events, and projections.
- Rebuild domain projections with `uv run victus projections-rebuild <user_id>` after event recovery.
- Do not fabricate checkpoints from legacy `pending_interaction_state`; ask the user to repeat it.
- Legacy summary/pending tables remain read-only compatibility data until a separately audited drop.

# Phoenix observability

For the shared local Compose stack, configure both the gateway and agent with:

```bash
PHOENIX_TRACING_ENABLED=true
PHOENIX_COLLECTOR_ENDPOINT=http://victus-phoenix:6006
PHOENIX_PROJECT_NAME=victus-local
OPENINFERENCE_HIDE_INPUTS=false
OPENINFERENCE_HIDE_OUTPUTS=false
OPENINFERENCE_HIDE_INPUT_MESSAGES=false
OPENINFERENCE_HIDE_OUTPUT_MESSAGES=false
```

Phoenix project `victus-local` shows one chat trace rooted at `webapp.chat.stream`, followed by
`gateway.agent.request`, `agent.http.chat`, and an `agent.turn` span of kind `AGENT`. Its readable
children are `agent.safety_precheck`, `agent.decision`, `tool.event_capture`, and
`agent.final_response`. Application spans show concise semantic inputs and outputs. Their LLM
children, such as `llm.agent_decision`, retain the exact provider-bound input JSON, OpenInference
messages, advertised tool schemas, invocation parameters, output message, tool calls, and token
usage. LangGraph auto-instrumentation is disabled deliberately: graph state is not replicated into
the operational trace because OpenTelemetry attributes are not an archival store.

# Troubleshooting

- `503 /health`: verify database reachability, LangGraph setup, and LiteLLM configuration.
- `401 /chat`: verify the bearer token against `BACKEND_API_URL/me`.
- `404 /chat/debug`: set `VICTUS_CHAT_DEBUG_ENABLED=true` and restart the chat service.
- `403 /chat`: the authenticated user does not own that conversation.
- `409 /chat`: resume was sent without a pending interrupt, message was sent while an interrupt was
  pending, or the checkpoint graph version is incompatible.
- `503 /chat`: inspect database and LiteLLM availability; responses intentionally hide provider and
  credential details.
