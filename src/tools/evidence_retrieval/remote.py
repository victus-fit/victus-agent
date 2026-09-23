from __future__ import annotations

from dataclasses import dataclass

import httpx

from tools.evidence_retrieval.contract import EvidenceRetrievalInput, EvidenceSearchResponse


class EvidenceRetrievalUnavailable(RuntimeError):
    pass


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
            return EvidenceSearchResponse.model_validate(response.json())
        except (httpx.HTTPStatusError, ValueError) as exc:
            raise EvidenceRetrievalUnavailable("evidence retrieval returned an invalid response") from exc
