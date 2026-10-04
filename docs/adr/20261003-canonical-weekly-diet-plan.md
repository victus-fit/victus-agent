# Canonical weekly diet-plan document

## Context

The diet-plan tool previously accepted an arbitrary JSON object. The model could
therefore activate incomplete revisions such as two unnamed snack entries, which
the weekly-plan UI cannot represent as a meaningful nutrition plan.

## Decision

`diet_plan` now accepts a structured document with daily macro targets and
exactly seven unique Spanish weekdays. Every weekday has two or three named
meals, and every meal has at least one named food and portion. The WebApp
validates the same invariant before a revision is persisted.

Diet-plan intake is a three-step state: preferences, meals per day, and cooking
time. A complete proposal is saved as a draft and shown for review. Only an
explicit user acceptance activates it.

## Consequences

Malformed tool calls are rejected rather than activated. Existing revisions are
unchanged. Draft refinements do not alter the active plan; activating the
accepted draft archives the prior active plan.
