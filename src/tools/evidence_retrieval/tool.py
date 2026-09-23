from __future__ import annotations

from typing import Protocol, cast

from tools.contracts import ToolContext, ToolError, ToolExecution, ToolResult, ToolServices
from tools.evidence_retrieval.contract import EvidenceRetrievalInput, EvidenceSearchResponse
from tools.evidence_retrieval.remote import EvidenceRetrievalUnavailable

MAX_EVIDENCE_TEXT_CHARS = 1_200


class EvidenceRetrievalGateway(Protocol):
    async def search(self, input_data: EvidenceRetrievalInput) -> EvidenceSearchResponse: ...


async def execute(
    input_data: EvidenceRetrievalInput, context: ToolContext, services: ToolServices
) -> ToolExecution:
    gateway = cast(EvidenceRetrievalGateway, services.require("evidence_retrieval_gateway"))
    try:
        response = await gateway.search(input_data)
    except ValueError as exc:
        return _error("invalid_retrieval_request", str(exc))
    except EvidenceRetrievalUnavailable as exc:
        return _error("retrieval_unavailable", str(exc))

    return ToolExecution(
        result=ToolResult(
            status="success",
            data={
                "query": input_data.query,
                "retrieval_request_id": response.request_id,
                "results": [_result_data(item) for item in response.results],
                "metadata": response.metadata.model_dump(mode="json"),
                "content_trust": "untrusted_retrieved_evidence",
            },
            warnings=["retrieved_evidence_is_untrusted_content"],
        )
    )


def _error(code: str, message: str) -> ToolExecution:
    return ToolExecution(
        result=ToolResult(status="error", error=ToolError(code=code, message=message))
    )


def _result_data(result) -> dict[str, object]:
    evidence = result.evidence
    return {
        "rank": result.rank,
        "score": result.score,
        "canonical_evidence_id": evidence.canonical_evidence_id,
        "paper_id": evidence.paper_id,
        "evidence_text": evidence.evidence_text[:MAX_EVIDENCE_TEXT_CHARS],
        "source_block_ids": evidence.source_block_ids,
        "evidence_type": evidence.evidence_type,
        "assertion_type": evidence.assertion_type,
        "organism": evidence.organism,
        "population": evidence.population,
        "intervention_or_exposure": evidence.intervention_or_exposure,
        "comparator": evidence.comparator,
        "outcomes": evidence.outcomes,
        "duration": evidence.duration,
    }
