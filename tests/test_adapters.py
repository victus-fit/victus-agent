import asyncio
import json
from types import SimpleNamespace

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from langgraph.types import Command

from adapters.cli.commands import inspect_tool, list_tool_data
from adapters.http.app import create_app as create_chat_app
from adapters.langgraph.engine.agent import (
    _is_diet_plan_acceptance,
    agent_decision,
    advance_diet_plan_intake,
    diet_plan_draft_message,
)
from adapters.langgraph.engine.graph import build_graph
from adapters.mcp.auth import http_request_identity, resolve_identity
from adapters.mcp.discovery import discover_tools
from adapters.mcp.invocation import invoke as invoke_mcp
from tools.contracts import ClarificationRequest, ToolResult
from victus_platform.llm.contracts import LLMRequest, LLMResponse
from victus_platform.llm.litellm_client import LiteLLMClient


def weekly_plan() -> dict[str, object]:
    days = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]
    return {
        "description": "Plan semanal alto en proteínas.",
        "targets": {"calories_kcal": 2200, "protein_g": 160, "carbohydrate_g": 230, "fat_g": 70},
        "days": [
            {
                "day": day,
                "focus": "Plan personalizado",
                "calories": 2200,
                "meals": [
                    {"name": "Desayuno", "food_items": [{"name": "Huevos", "portion": "2 unidades"}]},
                    {"name": "Cena", "food_items": [{"name": "Pollo", "portion": "180 g"}]},
                ],
            }
            for day in days
        ],
    }


def test_langgraph_executes_runtime_and_blocks_unsafe_turns() -> None:
    client = SequenceClient(
        [
            LLMResponse(
                text="",
                tool_calls=[
                    {
                        "name": "event_capture",
                        "arguments": {
                            "items": [{"name": "arroz", "quantity": 100, "unit": "g"}],
                        },
                    }
                ],
            ),
            LLMResponse(text="Registrado.", tool_calls=[]),
        ]
    )
    allowed = asyncio.run(
        build_graph(llm_client=client).ainvoke(
            {
                "request": {
                    "request_id": "r1",
                    "user_id": "u1",
                    "conversation_id": "c1",
                    "raw_text": "hoy comi arroz",
                }
            }
        )
    )
    assert allowed["tool_context"]["last_tool_result"]["data"]["capture_action"] == "log_meal"
    assert allowed["tool_context"]["allowed_tools"] == ["event_capture", "evidence_retrieval", "profile", "diet_plan"]
    assert [tool["function"]["name"] for tool in client.requests[0].tools or []] == [
        "event_capture",
        "evidence_retrieval",
        "profile",
        "diet_plan",
    ]
    assert allowed["audit"]["node_path"][-1] == "finalize_turn"
    assert "intent" not in allowed

    blocked = asyncio.run(
        build_graph(safety_client=BlockedSafetyClient()).ainvoke(
            {
                "request": {
                    "request_id": "r2",
                    "user_id": "u1",
                    "conversation_id": "c2",
                    "raw_text": "I am going to hurt myself",
                }
            }
        )
    )
    assert blocked["safety"]["status"] == "blocked"
    assert blocked["tool_context"]["allowed_tools"] == []
    assert "last_tool_result" not in blocked["tool_context"]
    assert "intent" not in blocked


def test_langgraph_model_selection_keeps_identity_and_text_out_of_tool_arguments() -> None:
    client = SequenceClient(
        [
            LLMResponse(
                text="",
                tool_calls=[
                    {
                        "id": "call-1",
                        "name": "event_capture",
                        "arguments": {"items": [{"name": "arroz", "quantity": 100, "unit": "g"}]},
                    }
                ],
            ),
            LLMResponse(text="Registrado.", tool_calls=[]),
        ]
    )
    runtime = RecordingRuntime()
    result = asyncio.run(
        build_graph(llm_client=client, tool_runtime=runtime).ainvoke(
            {
                "request": {
                    "request_id": "r1",
                    "user_id": "u1",
                    "conversation_id": "c1",
                    "raw_text": "Hoy comí arroz",
                }
            }
        )
    )
    assert runtime.invocations[0].arguments == {
        "items": [{"name": "arroz", "quantity": 100, "unit": "g"}]
    }
    assert runtime.invocations[0].context.original_text == "Hoy comí arroz"
    assert result["tool_context"]["last_tool_result"]["status"] == "success"
    assert result["response"]["user_message"] == "Registrado."
    assert "load_domain_projections" not in result["audit"]["node_path"]
    assert '"projections"' not in client.requests[0].messages[0]["content"]

    changed_identity = asyncio.run(
        build_graph(
            llm_client=SequenceClient(
                [
                    LLMResponse(
                        text="",
                        tool_calls=[
                            {
                                "name": "event_capture",
                                "arguments": {"user_id": "other", "items": [{"name": "arroz", "quantity": 100, "unit": "g"}]},
                            }
                        ],
                    )
                ]
            )
        ).ainvoke(
            {
                "request": {
                    "request_id": "r2",
                    "user_id": "u1",
                    "conversation_id": "c2",
                    "raw_text": "texto",
                }
            }
        )
    )
    assert changed_identity["response"]["mode"] == "error"
    assert "identity" in changed_identity["response"]["user_message"]


def test_diet_plan_intake_collects_two_answers_then_generates_a_draft() -> None:
    intake = {"status": "collecting", "step": "preferences", "answers": {}}
    intake, question = advance_diet_plan_intake(intake, "Sin gluten; me gustan la carne y los huevos.")
    assert question == "Perfecto. ¿Cuántas comidas al día quieres realizar?"
    assert intake["answers"]["preferences"].startswith("Sin gluten")

    intake, question = advance_diet_plan_intake(intake, "Tres comidas al día.")
    assert question is None
    assert intake["status"] == "ready"
    assert intake["answers"]["meals_per_day"] == "Tres comidas al día."
    client = SequenceClient([LLMResponse(text=json.dumps(weekly_plan()), tool_calls=[])])
    state = {
        "request": {"conversation_id": "c-diet", "request_id": "diet-3"},
        "planning": {"diet_plan_intake": {**intake, "profile": {"preferences": []}}},
        "tool_context": {"allowed_tools": ["diet_plan"], "loop_count": 0},
    }
    proposed = asyncio.run(agent_decision(llm_client=client, model="test")(state))
    assert client.requests[0].operation == "agent.diet_plan_draft"
    assert proposed["tool_context"]["proposed_action"]["arguments"]["action"] == "create"
    assert len(proposed["tool_context"]["proposed_action"]["arguments"]["plan_json"]["days"]) == 7


def test_diet_plan_draft_is_presented_for_acceptance_or_editing() -> None:
    message = diet_plan_draft_message(weekly_plan())
    assert "## Propuesta de plan semanal" in message
    assert "### Lunes" in message
    assert "**Desayuno:** Huevos (2 unidades)" in message
    assert "aún no está activo" in message
    assert "aceptarlo" in message and "editarlo" in message
    assert _is_diet_plan_acceptance("Sí, acepto el plan")
    assert not _is_diet_plan_acceptance("Cambia la cena del martes")


def test_successful_draft_creation_is_presented_without_another_generation() -> None:
    state = {
        "request": {"original_text": "3"},
        "planning": {"diet_plan_intake": {"status": "ready", "answers": {}}},
        "tool_context": {
            "allowed_tools": ["diet_plan"],
            "loop_count": 1,
            "proposed_action": {"tool_name": "diet_plan", "arguments": {"action": "create", "plan_json": weekly_plan()}},
            "last_tool_result": {"status": "success", "data": {"plan_id": "draft-1"}},
        },
    }
    result = asyncio.run(agent_decision(llm_client=None, model="test")(state))
    assert result["planning"]["diet_plan_intake"]["status"] == "awaiting_confirmation"
    assert result["planning"]["diet_plan_intake"]["draft_plan_id"] == "draft-1"
    assert "Propuesta de plan semanal" in result["response"]["user_message"]


def test_diet_plan_acceptance_activates_the_pending_draft_once() -> None:
    state = {
        "request": {"original_text": "Sí, acepto el plan"},
        "planning": {"diet_plan_intake": {"status": "awaiting_confirmation", "draft_plan_id": "draft-1"}},
        "tool_context": {"allowed_tools": ["diet_plan"], "loop_count": 0},
    }
    proposed = asyncio.run(agent_decision(llm_client=None, model="test")(state))
    assert proposed["tool_context"]["proposed_action"]["arguments"] == {"action": "activate", "plan_id": "draft-1"}

    completed = asyncio.run(
        agent_decision(llm_client=None, model="test")(
            {
                **proposed,
                "tool_context": {
                    **proposed["tool_context"],
                    "last_tool_result": {"status": "success", "data": {"plan_id": "draft-1"}},
                },
            }
        )
    )
    assert "diet_plan_intake" not in completed["planning"]
    assert "quedó activo" in completed["response"]["user_message"]


def test_langgraph_confirmation_resumes_once_and_memory_is_user_scoped() -> None:
    saver = InMemorySaver()
    store = InMemoryStore()
    runtime = RecordingRuntime()
    client = SequenceClient(
        [
            LLMResponse(
                text="",
                tool_calls=[
                    {
                        "id": "call-1",
                        "name": "planning",
                        "arguments": {"user_id": "u1", "action": "adjust_goal", "goal_id": "g1", "patch": {}},
                    }
                ],
            ),
            LLMResponse(text="Meta ajustada.", tool_calls=[]),
        ]
    )
    graph = build_graph(llm_client=client, tool_runtime=runtime, checkpointer=saver, store=store)
    config = {"configurable": {"thread_id": "c-confirm", "user_id": "u1"}}
    paused = asyncio.run(
        graph.ainvoke(
            {
                "request": {
                    "request_id": "r1",
                    "user_id": "u1",
                    "conversation_id": "c-confirm",
                    "raw_text": "ajusta mi meta",
                }
            },
            config=config,
        )
    )
    assert paused["__interrupt__"][0].value["kind"] == "confirmation"
    resumed = asyncio.run(graph.ainvoke(Command(resume={"accepted": True}), config=config))
    assert resumed["response"]["user_message"] == "Meta ajustada."
    assert len(runtime.invocations) == 1

    declined_runtime = RecordingRuntime()
    declined_graph = build_graph(
        llm_client=SequenceClient(
            [
                LLMResponse(
                    text="",
                    tool_calls=[
                        {
                            "name": "planning",
                            "arguments": {"user_id": "u1", "action": "end_session", "session_id": "s1"},
                        }
                    ],
                )
            ]
        ),
        tool_runtime=declined_runtime,
        checkpointer=InMemorySaver(),
    )
    declined_config = {"configurable": {"thread_id": "declined", "user_id": "u1"}}
    asyncio.run(
        declined_graph.ainvoke(
            {
                "request": {
                    "request_id": "declined",
                    "user_id": "u1",
                    "conversation_id": "declined",
                    "raw_text": "termina la sesión",
                }
            },
            config=declined_config,
        )
    )
    declined = asyncio.run(
        declined_graph.ainvoke(Command(resume={"accepted": False}), config=declined_config)
    )
    assert declined["response"]["user_message"] == "Acción cancelada."
    assert declined_runtime.invocations == []

    memory_graph = build_graph(checkpointer=InMemorySaver(), store=store)
    asyncio.run(
        memory_graph.ainvoke(
            {
                "request": {
                    "request_id": "remember",
                    "user_id": "u1",
                    "conversation_id": "memory-1",
                    "raw_text": "recuerda que prefiero respuestas breves",
                }
            },
            config={"configurable": {"thread_id": "memory-1", "user_id": "u1"}},
        )
    )
    same_user = asyncio.run(
        memory_graph.ainvoke(
            {
                "request": {
                    "request_id": "recall",
                    "user_id": "u1",
                    "conversation_id": "memory-2",
                    "raw_text": "hola",
                }
            },
            config={"configurable": {"thread_id": "memory-2", "user_id": "u1"}},
        )
    )
    other_user = asyncio.run(
        memory_graph.ainvoke(
            {
                "request": {
                    "request_id": "other",
                    "user_id": "u2",
                    "conversation_id": "memory-3",
                    "raw_text": "hola",
                }
            },
            config={"configurable": {"thread_id": "memory-3", "user_id": "u2"}},
        )
    )
    assert same_user["memory"]["recalled"][0]["content"] == "prefiero respuestas breves"
    assert other_user["memory"]["recalled"] == []


def test_langgraph_clarification_survives_checkpoint_resume() -> None:
    runtime = SequenceRuntime(
        [
            ToolResult(
                status="needs_clarification",
                clarification=ClarificationRequest(
                    missing_fields=["time"],
                    question="¿A qué hora?",
                    expected_answer_type="time",
                ),
            ),
            ToolResult(status="success"),
        ]
    )
    client = SequenceClient(
        [
            LLMResponse(text="", tool_calls=[{"name": "event_capture", "arguments": {"items": [{"name": "arroz"}]}}]),
            LLMResponse(
                text='{"items":[{"name":"arroz","quantity":150,"unit":"g"}],"occurred_at_text":"today"}'
            ),
            LLMResponse(text="Registrado con la hora.", tool_calls=[]),
        ]
    )
    graph = build_graph(llm_client=client, tool_runtime=runtime, checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "clarify", "user_id": "u1"}}
    paused = asyncio.run(
        graph.ainvoke(
            {
                "request": {
                    "request_id": "clarify-1",
                    "user_id": "u1",
                    "conversation_id": "clarify",
                    "raw_text": "comí arroz",
                }
            },
            config=config,
        )
    )
    assert paused["__interrupt__"][0].value["question"] == "¿A qué hora?"
    resumed = asyncio.run(graph.ainvoke(Command(resume={"answer": "150 g"}), config=config))
    assert resumed["response"]["user_message"] == "Registrado con la hora."
    assert runtime.invocations[-1].arguments == {
        "items": [{"name": "arroz", "quantity": 150, "unit": "g"}],
        "occurred_at_text": "today",
    }
    assert runtime.invocations[-1].context.original_text == "150 g"
    assert client.requests[1].operation == "agent.clarification_merge"
    final_messages = client.requests[2].messages
    tool_response = next(
        message
        for message in final_messages
        if message.get("role") == "tool" and message.get("tool_call_id") == "clarification_merge"
    )
    assistant_call = next(
        message
        for message in final_messages
        if message.get("role") == "assistant"
        and any(call.get("id") == "clarification_merge" for call in message.get("tool_calls", []))
    )
    assert tool_response["tool_call_id"] == assistant_call["tool_calls"][0]["id"]


def test_langgraph_preserves_backend_candidate_question_without_missing_fields() -> None:
    runtime = SequenceRuntime(
        [
            ToolResult(
                status="needs_clarification",
                clarification=ClarificationRequest(
                    missing_fields=[],
                    question="No puedo distinguir “pollo”. ¿Cuál fue: Pollo (Chicken) o Pollo asado (Roast chicken)?",
                    expected_answer_type="free_text",
                ),
            )
        ]
    )
    client = SequenceClient(
        [LLMResponse(text="", tool_calls=[{"name": "event_capture", "arguments": {"items": [{"name": "pollo", "quantity": 100, "unit": "g"}]}}])]
    )
    paused = asyncio.run(
        build_graph(llm_client=client, tool_runtime=runtime, checkpointer=InMemorySaver()).ainvoke(
            {
                "request": {
                    "request_id": "candidate-1",
                    "user_id": "u1",
                    "conversation_id": "candidate",
                    "raw_text": "Comí 100 g de pollo",
                }
            },
            config={"configurable": {"thread_id": "candidate", "user_id": "u1"}},
        )
    )
    assert paused["__interrupt__"][0].value["question"].startswith("No puedo distinguir “pollo”")


def test_chat_api_authenticates_and_enforces_thread_ownership() -> None:
    from starlette.testclient import TestClient

    graph = build_graph(checkpointer=InMemorySaver(), store=InMemoryStore())
    with TestClient(
        create_chat_app(
            graph=graph,
            identity_resolver=TokenIdentityResolver(),
            debug_enabled=True,
        )
    ) as client:
        first = client.post(
            "/chat",
            headers={"Authorization": "Bearer user-one"},
            json={"conversation_id": "owned", "request_id": "r1", "message": "hoy comi arroz"},
        )
        forbidden = client.post(
            "/chat",
            headers={"Authorization": "Bearer user-two"},
            json={"conversation_id": "owned", "request_id": "r2", "message": "hola"},
        )
        debug = client.post(
            "/chat/debug",
            headers={"Authorization": "Bearer user-one"},
            json={"conversation_id": "debug-owned", "request_id": "r3", "message": "hola"},
        )
        debug_forbidden = client.post(
            "/chat/debug",
            headers={"Authorization": "Bearer user-two"},
            json={"conversation_id": "debug-owned", "request_id": "r4", "message": "hola"},
        )
    assert first.status_code == 200
    assert first.json()["status"] == "completed"
    assert "debug" not in first.json()
    assert forbidden.status_code == 403
    assert debug.status_code == 200
    assert debug.json()["request_id"] == "r3"
    assert debug.json()["debug"]["authenticated_user_id"] == "u1"
    assert debug.json()["debug"]["state"]["audit"]["node_path"][-1] == "finalize_turn"
    assert debug_forbidden.status_code == 403


def test_chat_api_preserves_tool_calls_across_conversation_turns() -> None:
    from starlette.testclient import TestClient

    llm_client = SequenceClient(
        [
            LLMResponse(
                text="",
                tool_calls=[
                    {
                        "id": "call-1",
                        "name": "event_capture",
                        "arguments": {"user_id": "u1"},
                    }
                ],
            ),
            LLMResponse(text="Registrado.", tool_calls=[]),
            LLMResponse(text="¿Qué más necesitas?", tool_calls=[]),
        ]
    )
    graph = build_graph(
        llm_client=llm_client,
        tool_runtime=RecordingRuntime(),
        checkpointer=InMemorySaver(),
        store=InMemoryStore(),
    )
    with TestClient(
        create_chat_app(graph=graph, identity_resolver=TokenIdentityResolver())
    ) as client:
        first = client.post(
            "/chat",
            headers={"Authorization": "Bearer user-one"},
            json={"conversation_id": "tool-thread", "request_id": "r1", "message": "comí arroz"},
        )
        second = client.post(
            "/chat",
            headers={"Authorization": "Bearer user-one"},
            json={"conversation_id": "tool-thread", "request_id": "r2", "message": "hola"},
        )

    assert first.status_code == 200
    assert second.status_code == 200
    assert second.json()["message"] == "¿Qué más necesitas?"
    follow_up_messages = llm_client.requests[-1].messages
    assistant = next(message for message in follow_up_messages if message["role"] == "assistant")
    tool = next(message for message in follow_up_messages if message["role"] == "tool")
    assert assistant["tool_calls"][0]["id"] == "call-1"
    assert tool["tool_call_id"] == "call-1"


def test_chat_debug_is_opt_in_and_redacts_sensitive_state() -> None:
    from starlette.testclient import TestClient

    disabled_app = create_chat_app(
        graph=DebugStateGraph(),
        identity_resolver=TokenIdentityResolver(),
        debug_enabled=False,
    )
    with TestClient(disabled_app) as client:
        hidden = client.post(
            "/chat/debug",
            headers={"Authorization": "Bearer user-one"},
            json={"conversation_id": "debug", "request_id": "r1", "message": "hola"},
        )
    assert hidden.status_code == 404

    enabled_app = create_chat_app(
        graph=DebugStateGraph(),
        identity_resolver=TokenIdentityResolver(),
        debug_enabled=True,
    )
    with TestClient(enabled_app) as client:
        response = client.post(
            "/chat/debug",
            headers={"Authorization": "Bearer user-one"},
            json={"conversation_id": "debug", "request_id": "r1", "message": "hola"},
        )
    payload = response.json()
    assert response.status_code == 200
    assert payload["debug"]["next_nodes"] == ["confirmation_interrupt"]
    assert payload["debug"]["state"]["tool_context"]["api_key"] == "[redacted]"
    assert "do-not-return" not in response.text


def test_chat_auto_resumes_clarification_when_message_arrives_while_pending() -> None:
    from starlette.testclient import TestClient

    graph = PendingInterruptGraph()
    with TestClient(
        create_chat_app(
            graph=graph,
            identity_resolver=TokenIdentityResolver(),
            debug_enabled=True,
        )
    ) as client:
        response = client.post(
            "/chat/debug",
            headers={"Authorization": "Bearer user-one"},
            json={"conversation_id": "paused", "request_id": "r2", "message": "a las 13:00"},
        )

    assert response.status_code == 200
    assert graph.invoked is True
    assert graph.last_resume == {"answer": "a las 13:00"}


def test_mcp_discovers_catalog_and_serves_http_health() -> None:
    from starlette.requests import Request

    from adapters.mcp.transport import MCP_PATH, create_app

    discovered_tools = [tool.name for tool in discover_tools()]
    assert discovered_tools[0] == "event_capture"
    assert discovered_tools == ["event_capture", "evidence_retrieval", "profile"]
    app = create_app(storage_preparer=_prepare_nothing)
    health_route = next(route for route in app.routes if getattr(route, "path", None) == "/health")
    response = asyncio.run(
        health_route.endpoint(
            Request({"type": "http", "method": "GET", "path": "/health", "headers": []})
        )
    )
    assert response.status_code == 200
    assert b'"transport":"streamable_http"' in response.body
    assert MCP_PATH == "/mcp"


def test_mcp_reuses_chat_identity_resolver_per_http_request(monkeypatch) -> None:
    class Resolver:
        def __init__(self) -> None:
            self.tokens: list[str] = []

        async def resolve(self, token: str) -> str | None:
            self.tokens.append(token)
            return {"user-one": "u1", "user-two": "u2"}.get(token)

    resolver = Resolver()
    monkeypatch.setenv("VICTUS_API_TOKEN", "container-token")

    async def resolve_cases() -> tuple[object, object, object]:
        with http_request_identity("user-one"):
            first = await resolve_identity(resolver=resolver)
        with http_request_identity("user-two"):
            second = await resolve_identity(resolver=resolver)
        with http_request_identity(None):
            missing = await resolve_identity(resolver=resolver)
        return first, second, missing

    first, second, missing = asyncio.run(resolve_cases())
    assert first.subject == "u1"
    assert first.authenticated is True
    assert second.subject == "u2"
    assert second.authenticated is True
    assert missing.subject is None
    assert missing.authenticated is False
    assert resolver.tokens == ["user-one", "user-two"]


def test_mcp_invocation_passes_backend_resolved_identity_to_the_runtime() -> None:
    class Resolver:
        async def resolve(self, token: str) -> str | None:
            return "u1" if token == "user-one" else None

    runtime = RecordingRuntime()

    async def invoke_with_identity() -> ToolResult:
        with http_request_identity("user-one"):
            return await invoke_mcp(
                runtime,
                "event_capture",
                {"items": [{"name": "arroz", "quantity": 100, "unit": "g"}]},
                identity_resolver=Resolver(),
            )

    result = asyncio.run(invoke_with_identity())
    assert result.status == "success"
    assert runtime.invocations[0].context.identity.subject == "u1"
    assert runtime.invocations[0].context.identity.authenticated is True


async def _prepare_nothing() -> None:
    return None


def test_cli_reads_catalog_without_alternate_execution_metadata() -> None:
    assert list_tool_data()[0]["name"] == "event_capture"
    inspected = inspect_tool("event_capture")
    assert inspected["input_schema"]["title"] == "EventCaptureInput"


def test_litellm_proxy_alias_maps_to_openai_compatible_provider(monkeypatch) -> None:
    monkeypatch.setenv("LITELLM_PROXY_API_BASE", "http://litellm:4000/v1")
    kwargs = LiteLLMClient()._kwargs(
        LLMRequest(
            operation="test.proxy_alias",
            model="litellm_proxy/gemini-flash-lite",
            messages=[{"role": "user", "content": "hola"}],
        )
    )
    assert kwargs["model"] == "openai/gemini-flash-lite"
    assert kwargs["api_base"] == "http://litellm:4000/v1"


class BlockedSafetyClient:
    def complete(self, request: LLMRequest) -> LLMResponse:
        return LLMResponse(text="unsafe\nS11")

    async def acomplete(self, request: LLMRequest) -> LLMResponse:
        raise AssertionError("unexpected async safety call")


class SequenceClient:
    def __init__(self, responses: list[LLMResponse]) -> None:
        self.responses = responses
        self.requests: list[LLMRequest] = []

    def complete(self, request: LLMRequest) -> LLMResponse:
        raise AssertionError("unexpected sync model call")

    async def acomplete(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        return self.responses.pop(0)


class RecordingRuntime:
    def __init__(self) -> None:
        self.invocations = []

    async def invoke_async(self, invocation):
        self.invocations.append(invocation)
        return ToolResult(status="success")


class SequenceRuntime(RecordingRuntime):
    def __init__(self, results: list[ToolResult]) -> None:
        super().__init__()
        self.results = results

    async def invoke_async(self, invocation):
        self.invocations.append(invocation)
        return self.results.pop(0)


class TokenIdentityResolver:
    async def resolve(self, token: str) -> str | None:
        return {"user-one": "u1", "user-two": "u2"}.get(token)


class DebugStateGraph:
    def __init__(self):
        self.invoked = False

    async def aget_state(self, config):
        next_nodes = ("confirmation_interrupt",) if self.invoked else ()
        return SimpleNamespace(values={}, next=next_nodes)

    async def ainvoke(self, graph_input, *, config):
        self.invoked = True
        return {
            "graph_version": "1",
            "request": {
                "request_id": "r1",
                "user_id": config["configurable"]["user_id"],
                "conversation_id": config["configurable"]["thread_id"],
            },
            "response": {"mode": "final", "user_message": "Debug listo."},
            "tool_context": {"api_key": "do-not-return", "tool_results": []},
            "audit": {"node_path": ["finalize_turn"]},
        }


class PendingInterruptGraph:
    def __init__(self):
        self.invoked = False
        self.last_resume = None

    async def aget_state(self, config):
        return SimpleNamespace(
            values={
                "graph_version": "1",
                "request": {"user_id": config["configurable"]["user_id"]},
            },
            next=("clarification_interrupt",),
        )

    async def ainvoke(self, graph_input, *, config):
        self.invoked = True
        self.last_resume = graph_input.resume
        return {
            "graph_version": "1",
            "request": {
                "request_id": "r2",
                "user_id": config["configurable"]["user_id"],
                "conversation_id": config["configurable"]["thread_id"],
            },
            "response": {"mode": "final", "user_message": "Aclaración recibida."},
            "audit": {"node_path": ["finalize_turn"]},
        }
