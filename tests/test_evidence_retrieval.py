import asyncio
import json

import httpx

from adapters.langgraph.engine.graph import build_graph
from tools.contracts import ToolContext, ToolIdentity, ToolInvocation, ToolServices
from tools.evidence_retrieval.contract import EvidenceRetrievalInput
from tools.evidence_retrieval.remote import EvidenceRetrievalUnavailable, VictusRAGEvidenceGateway
from tools.evidence_retrieval.tool import execute
from tools.runtime import ToolRuntime
from victus_platform.llm.contracts import LLMRequest, LLMResponse


def test_gateway_calls_victus_rag_with_bearer_and_contract_payload() -> None:
    received = {}

    def handler(request: httpx.Request) -> httpx.Response:
        received["url"] = str(request.url)
        received["authorization"] = request.headers["Authorization"]
        received["body"] = json.loads(request.content)
        return httpx.Response(200, json=_response())

    gateway = VictusRAGEvidenceGateway(
        base_url="http://victus-rag:8080/",
        api_token="service-token",
        transport=httpx.MockTransport(handler),
    )
    response = asyncio.run(
        gateway.search(EvidenceRetrievalInput(query=" creatine strength ", top_k=2))
    )

    assert received == {
        "url": "http://victus-rag:8080/v1/evidence/search",
        "authorization": "Bearer service-token",
        "body": {"query": "creatine strength", "top_k": 2},
    }
    assert response.results[0].evidence.canonical_evidence_id == "evidence-1"


def test_gateway_maps_unavailable_response_without_exposing_token() -> None:
    gateway = VictusRAGEvidenceGateway(
        base_url="http://victus-rag:8080",
        api_token="service-token",
        transport=httpx.MockTransport(lambda request: httpx.Response(503)),
    )

    try:
        asyncio.run(gateway.search(EvidenceRetrievalInput(query="creatine")))
    except EvidenceRetrievalUnavailable as exc:
        assert str(exc) == "evidence retrieval is unavailable"
        assert "service-token" not in str(exc)
    else:  # pragma: no cover - makes the expected error explicit
        raise AssertionError("expected evidence retrieval to be unavailable")


def test_tool_bounds_evidence_and_preserves_citation_fields() -> None:
    runtime = ToolRuntime(
        services=ToolServices({"evidence_retrieval_gateway": FakeGateway(_response("x" * 1_300))})
    )
    result = asyncio.run(
        runtime.invoke_async(
            ToolInvocation(
                name="evidence_retrieval",
                arguments={"query": "Does creatine improve strength?"},
                context=ToolContext(
                    source="test", identity=ToolIdentity(subject="u1", authenticated=True)
                ),
            )
        )
    )

    data = result.data
    assert result.status == "success"
    assert result.events_emitted == []
    assert result.warnings == ["retrieved_evidence_is_untrusted_content"]
    assert data["content_trust"] == "untrusted_retrieved_evidence"
    assert data["results"][0]["canonical_evidence_id"] == "evidence-1"
    assert data["results"][0]["paper_id"] == "paper-1"
    assert len(data["results"][0]["evidence_text"]) == 1_200


def test_langgraph_selects_evidence_retrieval_and_composes_from_its_result() -> None:
    client = SequenceClient(
        [
            LLMResponse(
                text="",
                tool_calls=[
                    {
                        "id": "evidence-call",
                        "name": "evidence_retrieval",
                        "arguments": {"query": "Does creatine improve strength?"},
                    }
                ],
            ),
            LLMResponse(text="La evidencia recuperada apoya una mejora de fuerza.", tool_calls=[]),
        ]
    )
    runtime = ToolRuntime(
        services=ToolServices({"evidence_retrieval_gateway": FakeGateway(_response())})
    )
    result = asyncio.run(
        build_graph(llm_client=client, tool_runtime=runtime).ainvoke(
            {
                "request": {
                    "request_id": "evidence-1",
                    "user_id": "u1",
                    "conversation_id": "c1",
                    "raw_text": "La creatina mejora la fuerza?",
                }
            }
        )
    )

    assert result["tool_context"]["last_tool_result"]["tool_name"] == "evidence_retrieval"
    assert result["response"]["user_message"] == "La evidencia recuperada apoya una mejora de fuerza."
    assert [tool["function"]["name"] for tool in client.requests[0].tools] == [
        "event_capture",
        "evidence_retrieval",
    ]


class FakeGateway:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    async def search(self, input_data: EvidenceRetrievalInput):
        from tools.evidence_retrieval.contract import EvidenceSearchResponse

        return EvidenceSearchResponse.model_validate(self.payload)


class SequenceClient:
    def __init__(self, responses: list[LLMResponse]) -> None:
        self.responses = responses
        self.requests: list[LLMRequest] = []

    def complete(self, request: LLMRequest) -> LLMResponse:
        raise AssertionError("unexpected synchronous model call")

    async def acomplete(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        return self.responses.pop(0)


def _response(evidence_text: str = "Creatine improved strength.") -> dict:
    return {
        "request_id": "ret-1",
        "results": [
            {
                "rank": 1,
                "score": 0.9,
                "evidence": {
                    "canonical_evidence_id": "evidence-1",
                    "paper_id": "paper-1",
                    "evidence_text": evidence_text,
                    "source_block_ids": ["paper-1:block-1"],
                    "organism": "human",
                },
            }
        ],
        "metadata": {
            "index_version": "canonical-evidence-v1",
            "retrieval_pipeline": "dense_bge_m3_v1",
            "took_ms": 10,
        },
    }
