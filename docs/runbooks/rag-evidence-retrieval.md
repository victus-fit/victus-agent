---
id: VICTUS-RUNBOOK-RAG-EVIDENCE-RETRIEVAL
title: RAG Evidence Retrieval Tool
status: current
---

# RAG Evidence Retrieval Tool

The LangGraph agent calls the private `victus-rag` evidence API only for scientific-evidence
questions. The RAG service owns retrieval; the agent owns synthesis and final wording.

## Configuration

Set these values in the agent deployment secret store or untracked environment file:

```bash
VICTUS_RAG_API_URL=http://victus-rag:8080
VICTUS_RAG_API_TOKEN='<RAG_API_TOKEN>'
VICTUS_RAG_API_TIMEOUT_SECONDS=10
```

The URL must resolve on the private network shared with `victus-rag`. The token is sent only as a
Bearer header and must match the RAG service `RAG_API_TOKEN`.

## Readiness

Before enabling the tool, verify the RAG service with its own token:

```bash
curl --header "Authorization: Bearer $RAG_API_TOKEN" http://victus-rag:8080/readyz
```

Then run the focused agent tests:

```bash
uv run --extra test pytest tests/test_evidence_retrieval.py tests/test_adapters.py -q
```

## Failure and rollback

`401`, `503`, timeouts, and malformed RAG responses become a normal `retrieval_unavailable` tool
error; chat remains available. To immediately disable retrieval without changing deployment
configuration, remove `evidence_retrieval` from `AGENT_ENABLED_TOOLS` and redeploy. Rotate the
shared token in both services if it is exposed.
