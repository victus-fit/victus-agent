---
id: ADR-20260829-HUMAN-READABLE-APPLICATION-SPANS
title: Human-Readable Application Spans With Raw LLM And Graph Diagnostics
status: accepted
date: 2026-08-29
owners:
  - victus-agent-runtime
---

# Context

Automatic LangGraph instrumentation preserves complete state and tool payloads, but those JSON
documents are not an effective first view of a chat turn in Phoenix.

# Decision

- Keep existing OpenInference LLM spans exhaustive: prompts, schemas, tool calls, provider output,
  and token usage remain available for local debugging.
- Disable automatic LangGraph instrumentation because its node and state spans obscure the execution
  narrative in Phoenix.
- Add manual application spans for `agent.turn`, safety, decision, tool execution, and final
  response. Their input/output values are human-readable semantic summaries rather than graph state.
- Mark `agent.turn` as OpenInference `AGENT`, application steps as `CHAIN`, application tools as
  `TOOL`, and provider calls as `LLM`.
- Preserve exhaustive provider-bound prompts, schemas, tool calls, raw responses, and tokens on the
  existing manual LLM spans. Do not duplicate unbounded graph-state JSON into application spans.
- Prefer the active OpenTelemetry application span when a LiteLLM call is started, so the LLM span is
  nested below its semantic decision span rather than a framework-owned span.

# Consequences

Phoenix can show an operationally readable tree while retaining the provider-level material needed
to investigate a model decision. OpenTelemetry attributes are not an archival store; preserving every
graph-state snapshot byte-for-byte would require a separately designed diagnostic-artifact store.
User-content retention remains governed by the existing local-debug tracing policy.
