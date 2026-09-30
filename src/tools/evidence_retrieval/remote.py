from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx

from tools.evidence_retrieval.contract import EvidenceRetrievalInput, EvidenceSearchResponse


class EvidenceRetrievalUnavailable(RuntimeError):
    pass


def _minimal_response(payload: object) -> dict[str, Any]:
    """Keep only the evidence text and source fields owned by this boundary."""
    if not isinstance(payload, dict):
        raise ValueError("response must be an object")
    results = payload.get("results")
    metadata = payload.get("metadata")
    if not isinstance(results, list) or not isinstance(metadata, dict):
        raise ValueError("response must contain results and metadata")
    projected_results: list[dict[str, Any]] = []
    for result in results:
        if not isinstance(result, dict) or not isinstance(result.get("evidence"), dict):
            raise ValueError("each result must contain evidence")
        evidence = result["evidence"]
        projected_results.append(
            {
                "rank": result.get("rank"),
                "score": result.get("score"),
                "evidence": {
                    "canonical_evidence_id": evidence.get("canonical_evidence_id"),
                    "paper_id": evidence.get("paper_id"),
                    "paper_title": evidence.get("paper_title"),
                    "evidence_text": evidence.get("evidence_text"),
                    "source_block_ids": evidence.get("source_block_ids", []),
                },
            }
        )
    return {
        "request_id": payload.get("request_id"),
        "results": projected_results,
        "metadata": {
            "index_version": metadata.get("index_version"),
            "retrieval_pipeline": metadata.get("retrieval_pipeline"),
            "took_ms": metadata.get("took_ms"),
        },
    }


@dataclass(frozen=True)
class VictusRAGEvidenceGateway:
    base_url: str = ""
    api_token: str = ""
    timeout_seconds: float = 10.0
    transport: httpx.AsyncBaseTransport | None = None

    async def search(self, input_data: EvidenceRetrievalInput) -> EvidenceSearchResponse:
        if not self.base_url or not self.api_token:
            raise EvidenceRetrievalUnavailable("evidence retrieval is not configured")
        try:
            async with httpx.AsyncClient(
                timeout=self.timeout_seconds,
                transport=self.transport,
            ) as client:
                response = await client.post(
                    f"{self.base_url.rstrip('/')}/v1/evidence/search",
                    headers={"Authorization": f"Bearer {self.api_token}"},
                    json=input_data.as_request_payload(),
                )
        except httpx.TimeoutException as exc:
            raise EvidenceRetrievalUnavailable("evidence retrieval timed out") from exc
        except httpx.HTTPError as exc:
            raise EvidenceRetrievalUnavailable("evidence retrieval is unavailable") from exc

        if response.status_code == 401:
            raise EvidenceRetrievalUnavailable("evidence retrieval authentication failed")
        if response.status_code == 422:
            raise ValueError("evidence retrieval rejected the search request")
        if response.status_code >= 500:
            raise EvidenceRetrievalUnavailable("evidence retrieval is unavailable")
        try:
            response.raise_for_status()
            return EvidenceSearchResponse.model_validate(_minimal_response(response.json()))
        except (httpx.HTTPStatusError, ValueError) as exc:
            raise EvidenceRetrievalUnavailable("evidence retrieval returned an invalid response") from exc
