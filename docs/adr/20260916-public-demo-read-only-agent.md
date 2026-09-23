---
id: VICTUS-ADR-PUBLIC-DEMO-READ-ONLY
status: accepted
date: 2026-09-16
---

# Public demo uses the isolated ephemeral runtime profile

## Context

The public site needs an interactive demo for the immutable `demo:david` base profile that behaves
like the authenticated agent. A separate one-shot LLM path was safe, but diverged from production:
it had no LangGraph safety flow, tool selection, clarifications, RAG, or representative Phoenix trace.

## Decision

- Expose `POST /demo/chat` as a separate path with its own ES256 JWT verifier.
- Verify an embedded, configured public JWKS by `kid`; the agent never receives a signing key.
- Require issuer `victus-webapp`, a 30-second-or-less token lifetime, audience `victus-agent`, exact
  subject `demo:david`, `demo: true`, scopes `demo:chat` and `demo:read`, a versioned profile, a
  signed session identifier (`sid`), and single-use `jti`.
- Load `src/demo/fixtures/david-v1.json` from the deployed artifact. It is repository-editable but
  immutable at runtime. A future fixture is a new version, such as `david-v2`.
- Invoke the same LangGraph topology as `/chat`, with the normal safety precheck, LLM decision,
  clarification flow, canonical tools, response composition, and Phoenix spans.
- The `demo` execution profile permits only explicitly mapped tools: `event_capture` persists into a
  session-local in-memory event sink; `evidence_retrieval` calls the read-only Victus RAG gateway.
  Product databases, profile gateways, external actions, and every future tool are denied by default.
- Each signed session gets independent in-memory LangGraph checkpoints, memory, and virtual events.
  They expire after `VICTUS_DEMO_SESSION_TTL_SECONDS` (15 minutes by default) and are never written
  to PostgreSQL.
- Emit the same telemetry content and span structure as production, tagged with `victus.demo=true`,
  `victus.execution_mode=demo`, and fixture version. Phoenix is therefore an internal trusted system.

## Consequences

The fixture is immutable, while a visitor can accumulate temporary context within one signed session.
Refreshing the public page must mint a new `sid`, which starts from `david-v1` again. The in-process
session and replay caches are valid for the current single-process deployment; horizontal scaling
requires shared TTL-backed implementations before adding another agent replica.
