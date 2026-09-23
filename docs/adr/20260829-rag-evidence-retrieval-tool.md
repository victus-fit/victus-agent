---
id: ADR-20260829-RAG-EVIDENCE-RETRIEVAL-TOOL
title: Victus RAG Evidence Retrieval As A Read-Only Agent Tool
status: accepted
date: 2026-08-29
owners:
  - victus-agent-runtime
---

# Context

`victus-rag` provides private authenticated canonical-evidence retrieval but does not generate
answers. The agent needs that evidence inside its existing tool loop without taking ownership of
Qdrant, embeddings, or RAG credentials.

# Decision

- Add `evidence_retrieval` as a canonical asynchronous, read-only tool exposed only to LangGraph
  and tests.
- Call `POST /v1/evidence/search` with the RAG service bearer token held in agent configuration.
- Return bounded, cited evidence to the model and label it as untrusted retrieved content.
- Keep final synthesis in the agent and leave MCP/CLI exposure out of this first integration.

# Consequences

- Retrieval failures are normal tool errors and do not make chat unavailable.
- The tool can be disabled by removing it from the LangGraph allowlist.
- The agent and RAG service must share a private network and a rotated service token.
