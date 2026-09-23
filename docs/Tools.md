# Implemented Tools

## Tool Model

Every tool has a typed input contract and one implementation. The canonical catalog registers its
name, purpose, risk, side effects, and supported interfaces. `ToolRuntime` validates and executes the
same implementation for every adapter.

Canonical registry: `src/tools/catalog.py`

## Tools

- `event_capture` — Captures meals and beverages from food items with required numeric quantities
  in grams (`g`) or milliliters (`ml`); occurrence time defaults to today. Path:
  `src/tools/event_capture/tool.py`
- `evidence_retrieval` — Retrieves bounded, traceable scientific evidence from the private
  `victus-rag` API for LangGraph synthesis. It is read-only and initially unavailable through MCP
  and CLI. Path: `src/tools/evidence_retrieval/tool.py`

## Tool Result Contract

Every tool returns the same envelope. Tool-specific output belongs in `data`; execution status,
persisted event references, safety, tracing, and errors remain stable across every adapter.

Source: `src/tools/contracts.py`

```ts
type ToolStatus =
  | "success"
  | "needs_clarification"
  | "blocked"
  | "rejected"
  | "error"

type ToolEventRef = {
  event_id: string
  event_type: string
  seq: number
}

type ClarificationRequest = {
  missing_fields: string[]
  question: string
  expected_answer_type:
    | "quantity"
    | "time"
    | "meal_reference"
    | "preference_strength"
    | "restriction_type"
    | "goal_target"
    | "yes_no"
    | "free_text"
  resume_node?: string
  resume_action?: string
}

type ToolResult = {
  status: ToolStatus
  data?: unknown
  events_emitted: ToolEventRef[]
  warnings: string[]
  clarification?: ClarificationRequest
  safety: {
    status: "ok" | "warning" | "blocked" | "needs_clarification"
    reasons: string[]
  }
  error?: {
    code: string
    message: string
  }
}
```

Rules:

- `success` means the requested action completed.
- `needs_clarification` means execution must pause for missing information.
- `blocked` means safety or policy prevented execution.
- `rejected` means the invocation was invalid.
- `error` means execution failed unexpectedly.
- `events_emitted` contains references only after durable persistence.
- Unvalidated model output must never become persisted tool state.

## Demo execution profile

The public demo uses the same LangGraph tool catalog with an explicit allowlist. `event_capture`
writes only to an in-memory session event sink; `evidence_retrieval` remains a read-only RAG call.
No other tool is implicitly enabled for demo. See
[Public-Demo.md](contracts/agent/Public-Demo.md) for the authorization and TTL boundary.
