---
id: ADR-20260930-AGENT-MEAL-LOG-WEBAPP-PROJECTION
title: Agent meal capture writes to the webapp meal log
status: accepted
date: 2026-09-30
owners:
  - victus-agent-runtime
  - victus-webapp
---

# Context

The agent event store and the webapp meal-log table are separate stores. Agent-captured meals therefore did not appear in `/app`.

# Decision

For authenticated application users, `event_capture` calls a private backend endpoint. The backend resolves food names against its catalog and writes the canonical `user_meal_log_entries` rows consumed by `/app`. The public demo keeps its isolated temporary gateway.

# Consequences

- Chat and the meal-log workspace share one visible source of truth.
- Unknown or ambiguous foods return the existing clarification flow.
- The agent event store is no longer the delivery path for application meal logs.
