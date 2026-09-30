---
id: VICTUS-ADR-PROFILE-READ-TOOL
status: accepted
date: 2026-09-28
---

# Profile reads use the WebApp-owned persisted data boundary

## Context

The public demo previously serialized the repository fixture for `david-v1` into every agent
decision prompt. David is now initialized as a persisted WebApp user, so the fixture duplicates
the source of truth and causes unnecessarily broad prompt context.

## Decision

- Add the read-only canonical `profile` tool with sections `current_diet`, `biometrics`, and
  `overview`.
- The agent calls a WebApp internal endpoint using the existing service token. The WebApp owns the
  PostgreSQL queries and exposes only the latest logged diet and the latest values of weight,
  sleep, energy, and adherence.
- The endpoint maps exact subject `demo:david` to the seeded David user ID. Other callers must
  provide a UUID identity established by the authenticated agent request; no tool argument can
  override the subject.
- Remove the repository JSON fixture and require the LLM to invoke `profile` instead of receiving
  profile data eagerly.

## Consequences

Demo and authenticated users use the same read boundary while preserving demo write isolation.
Absent meal logs return an empty current diet rather than fabricated meals. Availability of the
WebApp endpoint and its shared service token is now required for profile answers.
