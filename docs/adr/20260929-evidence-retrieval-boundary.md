# Evidence retrieval boundary projection

## Context

Victus RAG can add retrieval or experiment metadata to evidence results. The
agent only needs evidence text and stable source identifiers to answer and cite
the user safely.

## Decision

At the agent gateway, project every RAG response into a minimal contract before
validation: evidence text, canonical evidence ID, paper ID, source block IDs,
and the transport metadata needed for tracing. Unknown RAG fields are ignored.
The projected contract remains strict: malformed result shape, missing text, or
missing canonical evidence ID fail the tool request.

## Consequences

RAG can evolve without breaking user-facing evidence retrieval. The agent does
not receive or expose unused retrieval payload fields. Contract tests use the
current RAG extra fields to preserve this behavior.
