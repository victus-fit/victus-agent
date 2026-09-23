---
id: VICTUS-ARCH-LANGGRAPH-FLOW
title: Current LangGraph Node Flow
status: current
updated_at: 2026-07-22
owners:
  - victus-agent-runtime
related_docs:
  - Overview.md
  - contracts/agent/Graph-State.md
related_modules:
  - src/adapters/langgraph/engine/graph.py
---

# Current LangGraph Node Flow

This document describes the nodes compiled by `build_graph()` for the authenticated chat runtime.
`graph.py` remains the executable source of truth; this is the human-readable map of its current
routes and node responsibilities.

## Routes

```text
START
  -> ingest_turn
  -> normalize_request
  -> recall_long_term_memory
  -> safety_precheck
     -> safety_blocked_response -> update_long_term_memory
     -> tool_registry -> agent_decision
        -> compose_final_response
        -> confirmation_interrupt -> compose_final_response | execute_tool
        -> execute_tool -> clarification_interrupt -> agent_decision
                        -> agent_decision
                        -> compose_final_response
  -> update_long_term_memory
  -> finalize_turn
  -> END
```

`Command(resume=...)` resumes the interrupted node in an existing checkpoint. It does not begin a
new route at `ingest_turn`.

## Nodes

- `ingest_turn`
  Module: `engine/agent.py`.
  Route: `START -> normalize_request`.
  Validates the authenticated thread identity and graph version; creates the user message and resets
  per-turn response and tool state.

- `normalize_request`
  Module: `runtime/context.py`.
  Route: `ingest_turn -> recall_long_term_memory`.
  Preserves the original request text and derives normalized working text.

- `recall_long_term_memory`
  Module: `runtime/memory.py`.
  Route: `normalize_request -> safety_precheck`.
  Loads bounded procedural and semantic memory for the authenticated user.

- `safety_precheck`
  Module: `runtime/context.py`.
  Route: `recall_long_term_memory -> safety_blocked_response | tool_registry`.
  Assesses whether the request can continue and selects the blocked or allowed route.

- `safety_blocked_response`
  Module: `runtime/context.py`.
  Route: `safety_precheck[blocked] -> update_long_term_memory`.
  Produces the bounded response for a safety block and removes available tools.

- `tool_registry`
  Module: `runtime/context.py`.
  Route: `safety_precheck[allowed] -> agent_decision`.
  Enables `event_capture` and the read-only `evidence_retrieval` tool for LangGraph model
  selection. The graph receives this allowlist from its execution profile; production and demo both
  currently expose these two tools, while demo substitutes ephemeral adapters for stateful effects.
  `evidence_retrieval` is intentionally inactive in MCP and CLI in this first release.

- `agent_decision`
  Module: `engine/agent.py`.
  Route: `tool_registry | clarification_interrupt | execute_tool[success] -> compose_final_response | confirmation_interrupt | execute_tool`.
  Uses the model to choose one allowed tool or a final answer; validates the proposed tool call and
  enforces the loop limit.

- `confirmation_interrupt`
  Module: `engine/agent.py`.
  Route: `agent_decision[confirm] -> compose_final_response | execute_tool`.
  Pauses for explicit approval of an action that requires confirmation.

- `execute_tool`
  Module: `engine/agent.py`.
  Route: `agent_decision | confirmation_interrupt -> clarification_interrupt | agent_decision | compose_final_response`.
  Invokes the selected canonical tool with authenticated context and stores its result.

- `clarification_interrupt`
  Module: `engine/agent.py`.
  Route: `execute_tool[needs_clarification] -> agent_decision`.
  Pauses for missing user information; on resume, makes the answer the next working request text.

- `compose_final_response`
  Module: `engine/agent.py`.
  Route: `agent_decision[response] | confirmation_interrupt[declined] | execute_tool[blocked/error] -> update_long_term_memory`.
  Converts a terminal graph or tool outcome into the bounded user-facing response.

- `update_long_term_memory`
  Module: `runtime/memory.py`.
  Route: `safety_blocked_response | compose_final_response -> finalize_turn`.
  Applies explicit remember/forget requests that pass the memory policy.

- `finalize_turn`
  Module: `engine/agent.py`.
  Route: `update_long_term_memory -> END`.
  Bounds checkpoint message and audit collections before the turn completes.

## Routing Semantics

- `safety_precheck` branches on the safety decision: `blocked` or `allowed`.
- `agent_decision` branches on a direct response, a confirmation requirement, or tool execution.
- `confirmation_interrupt` routes a declined confirmation to response composition; an accepted one
  executes the proposed tool.
- `execute_tool` routes a `needs_clarification` result to its interrupt, a `success` result back to
  the model for final wording, and blocked or error results to response composition.
- A clarification interrupt is part of the existing conversation checkpoint. The next browser
  request must use the chat resume contract rather than create a new message turn.

## Boundaries

Nodes coordinate state only. Canonical tool contracts and persistence live outside the graph:

- Tools execute through `ToolRuntime`.
- A tool loads a domain read model only when its implementation explicitly requires one; the graph
  does not preload projections for model selection.
- Events and projections remain the durable domain source of truth.
- The HTTP adapter authenticates the request and owns conversation access before graph invocation.
- `VictusGraphState` defines the shared state shape in
  [Graph-State.md](contracts/agent/Graph-State.md).
