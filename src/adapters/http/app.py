from __future__ import annotations

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import date, datetime, timezone
from time import perf_counter
from typing import Any

from langgraph.types import Command
from pydantic import BaseModel, ValidationError
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from demo.auth import (
    DemoAuthConfigurationError,
    DemoTokenError,
    DemoTokenVerifier,
)
from demo.contracts import DemoChatRequest, DemoChatResponse
from demo.fixture import load_david_fixture
from demo.policy import denied_demo_message, is_disallowed_demo_input
from demo.runtime import DemoRuntimeConfigurationError, DemoSessionManager
from adapters.http.auth import BackendIdentityResolver, IdentityResolver
from adapters.langgraph.capabilities.contracts import (
    ChatDebugExecution,
    ChatDebugResponse,
    ChatDebugSnapshot,
    ChatInterrupt,
    ChatRequest,
    ChatResponse,
)
from adapters.langgraph.engine.graph import build_graph
from adapters.langgraph.engine.state import GRAPH_VERSION
from adapters.langgraph.runtime.persistence import postgres_graph_resources
from bootstrap.storage import prepare_agent_storage
from victus_platform.database.engine import database_url
from victus_platform.llm.factory import build_llm_client
from victus_platform.telemetry.phoenix import (
    initialize_phoenix,
    phoenix_context,
    record_application_output,
    shutdown_phoenix,
    trace_application_span,
    trace_chat_request,
)

DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8766
DEBUG_STATE_FIELDS = (
    "request",
    "safety",
    "tool_context",
    "planning",
    "evidence",
    "clarification",
    "response",
    "memory",
    "audit",
    "messages",
)
SENSITIVE_FIELD_PARTS = (
    "api_key",
    "apikey",
    "authorization",
    "cookie",
    "credential",
    "password",
    "secret",
    "token",
)
MAX_DEBUG_DEPTH = 8
MAX_DEBUG_ITEMS = 100
MAX_DEBUG_STRING = 8_000


def create_app(
    *,
    graph: Any | None = None,
    identity_resolver: IdentityResolver | None = None,
    debug_enabled: bool | None = None,
    demo_session_manager: DemoSessionManager | None = None,
    demo_token_verifier: DemoTokenVerifier | None = None,
) -> Starlette:
    resolver = identity_resolver or BackendIdentityResolver()
    demo_configuration_error = False
    try:
        demo_sessions = demo_session_manager or DemoSessionManager(llm_client=build_llm_client())
    except DemoRuntimeConfigurationError:
        demo_sessions = None
        demo_configuration_error = True
    demo_fixture = load_david_fixture()
    demo_auth_configuration_error = False
    try:
        verifier = demo_token_verifier or DemoTokenVerifier.from_environment()
    except DemoAuthConfigurationError:
        verifier = None
        demo_auth_configuration_error = True
    debug_route_enabled = (
        _environment_flag("VICTUS_CHAT_DEBUG_ENABLED") if debug_enabled is None else debug_enabled
    )

    @asynccontextmanager
    async def lifespan(app: Starlette) -> AsyncIterator[None]:
        telemetry = initialize_phoenix()
        try:
            if graph is not None:
                app.state.graph = graph
                app.state.store = None
                yield
                return
            await prepare_agent_storage()
            async with postgres_graph_resources(database_url()) as resources:
                client = build_llm_client()
                app.state.graph = build_graph(
                    llm_client=client,
                    checkpointer=resources.checkpointer,
                    store=resources.store,
                )
                app.state.store = resources.store
                yield
        finally:
            shutdown_phoenix(telemetry)

    async def health(request: Request) -> JSONResponse:
        configured = bool(os.getenv("DATABASE_URL") and os.getenv("LITELLM_PROXY_API_BASE"))
        ready = configured or graph is not None
        if configured and graph is None:
            try:
                config = {
                    "configurable": {
                        "thread_id": "__readiness__",
                        "user_id": "__readiness__",
                    }
                }
                await request.app.state.graph.aget_state(config)
                await request.app.state.store.asearch(("system", "readiness"), limit=1)
            except Exception:
                ready = False
        return JSONResponse(
            {"status": "ok" if ready else "not_ready", "service": "victus-agent-chat"},
            status_code=200 if ready else 503,
        )

    async def handle_chat(request: Request, *, include_debug: bool) -> JSONResponse:
        token = _bearer_token(request)
        if token is None:
            return JSONResponse({"error": "missing bearer token"}, status_code=401)
        try:
            user_id = await resolver.resolve(token)
        except RuntimeError as exc:
            return JSONResponse({"error": str(exc)}, status_code=503)
        if not user_id:
            return JSONResponse({"error": "invalid bearer token"}, status_code=401)
        try:
            payload = ChatRequest.model_validate(await request.json())
        except (ValidationError, ValueError) as exc:
            return JSONResponse({"error": "invalid request", "details": str(exc)}, status_code=422)

        config = {"configurable": {"thread_id": payload.conversation_id, "user_id": user_id}}
        snapshot = await request.app.state.graph.aget_state(config)
        values = snapshot.values or {}
        owner = values.get("request", {}).get("user_id")
        if owner and owner != user_id:
            return JSONResponse({"error": "conversation is owned by another user"}, status_code=403)
        if values.get("graph_version") not in (None, GRAPH_VERSION):
            return JSONResponse({"error": "conversation graph version is incompatible"}, status_code=409)

        if payload.resume is not None:
            if not values or not snapshot.next:
                return JSONResponse({"error": "conversation is not awaiting a response"}, status_code=409)
            graph_input: Any = Command(resume=payload.resume.value)
        else:
            if snapshot.next:
                pending_nodes = [str(node) for node in snapshot.next]
                if pending_nodes == ["clarification_interrupt"]:
                    graph_input = Command(resume={"answer": payload.message})
                else:
                    return JSONResponse(
                        {
                            "error": "conversation is awaiting a response",
                            "message": "Resume the pending interrupt instead of sending a new message.",
                            "pending_nodes": pending_nodes,
                        },
                        status_code=409,
                    )
            else:
                graph_input = {
                    "request": {
                        "request_id": payload.request_id,
                        "user_id": user_id,
                        "raw_text": payload.message,
                        "conversation_id": payload.conversation_id,
                        "locale": payload.locale,
                        "timezone": payload.timezone,
                    }
                }

        started_at = datetime.now(timezone.utc)
        started_counter = perf_counter()
        try:
            with trace_chat_request(
                headers=dict(request.headers),
                attributes={
                    "victus.conversation_id": payload.conversation_id,
                    "victus.request_id": payload.request_id,
                    "victus.resumed": payload.resume is not None,
                    "victus.graph.version": GRAPH_VERSION,
                },
            ):
                with phoenix_context(
                    session_id=payload.conversation_id,
                    user_id=user_id,
                    metadata={
                        "request_id": payload.request_id,
                        "resumed": payload.resume is not None,
                        "graph_version": GRAPH_VERSION,
                    },
                ):
                    with trace_application_span(
                        "agent.turn",
                        input_value=payload.message,
                        attributes={
                            "victus.conversation_id": payload.conversation_id,
                            "victus.request_id": payload.request_id,
                            "victus.resumed": payload.resume is not None,
                        },
                        span_kind="AGENT",
                    ) as span:
                        result = await request.app.state.graph.ainvoke(graph_input, config=config)
                        response_state = result.get("response", {})
                        response_message = str(response_state.get("user_message") or "")
                        record_application_output(
                            span,
                            response_message or "Turn completed",
                            {"victus.response.mode": str(response_state.get("mode") or "unknown")},
                        )
        except Exception as exc:
            body: dict[str, Any] = {"error": "agent execution failed"}
            if include_debug:
                body["debug"] = {
                    "stage": "graph_execution",
                    "error_type": exc.__class__.__name__,
                }
            return JSONResponse(body, status_code=503)
        completed_at = datetime.now(timezone.utc)
        duration_ms = max(0, round((perf_counter() - started_counter) * 1_000))
        response = _chat_response(payload.conversation_id, result)
        if not include_debug:
            return JSONResponse(response.model_dump(mode="json", exclude_none=True))

        next_nodes: list[str] = []
        try:
            completed_snapshot = await request.app.state.graph.aget_state(config)
            next_nodes = [str(node) for node in (completed_snapshot.next or ())]
        except Exception:
            next_nodes = []
        debug_response = ChatDebugResponse(
            **response.model_dump(mode="python"),
            request_id=payload.request_id,
            debug=ChatDebugSnapshot(
                graph_version=str(result.get("graph_version") or "") or None,
                thread_id=payload.conversation_id,
                authenticated_user_id=user_id,
                next_nodes=next_nodes,
                execution=ChatDebugExecution(
                    started_at=started_at,
                    completed_at=completed_at,
                    duration_ms=duration_ms,
                    resumed=payload.resume is not None,
                ),
                state=_debug_state(result),
            ),
        )
        return JSONResponse(debug_response.model_dump(mode="json", exclude_none=True))

    async def chat(request: Request) -> JSONResponse:
        return await handle_chat(request, include_debug=False)

    async def chat_debug(request: Request) -> JSONResponse:
        if not debug_route_enabled:
            return JSONResponse({"error": "not found"}, status_code=404)
        return await handle_chat(request, include_debug=True)

    async def demo_chat(request: Request) -> JSONResponse:
        token = _bearer_token(request)
        if token is None:
            return JSONResponse({"error": "missing bearer token"}, status_code=401)
        if (
            demo_auth_configuration_error
            or demo_configuration_error
            or verifier is None
            or demo_sessions is None
        ):
            return JSONResponse({"error": "demo unavailable"}, status_code=503)
        try:
            identity = verifier.verify(token, profile_version=demo_fixture.profile_version)
        except DemoTokenError as exc:
            return JSONResponse({"error": "demo authorization failed"}, status_code=exc.status_code)
        try:
            payload = DemoChatRequest.model_validate(await request.json())
        except (ValidationError, ValueError):
            return JSONResponse({"error": "invalid request"}, status_code=422)

        try:
            with trace_application_span(
                "agent.turn",
                input_value=payload.message,
                attributes={
                    "victus.demo": True,
                    "victus.demo.fixture_version": identity.profile_version,
                    "victus.execution_mode": "demo",
                },
                span_kind="AGENT",
            ) as span:
                with phoenix_context(
                    session_id=identity.session_id,
                    user_id=identity.subject,
                    metadata={
                        "fixture_version": identity.profile_version,
                        "execution_mode": "demo",
                    },
                ):
                    with trace_application_span(
                        "demo.policy_precheck",
                        input_value=payload.message,
                        attributes={"victus.demo": True},
                        span_kind="CHAIN",
                    ) as policy_span:
                        policy_denied = is_disallowed_demo_input(payload.message)
                        record_application_output(
                            policy_span,
                            "Denied" if policy_denied else "Allowed",
                            {"victus.demo.policy_denied": policy_denied},
                        )
                    if policy_denied:
                        response = DemoChatResponse(
                            message=denied_demo_message(payload.language),
                            profile_version=demo_fixture.profile_version,
                        )
                    else:
                        result = await demo_sessions.invoke(
                            session_id=identity.session_id,
                            graph_input={
                                "request": {
                                    "request_id": payload.request_id,
                                    "user_id": f"demo:david:{identity.session_id}",
                                    "raw_text": payload.message,
                                    "conversation_id": identity.session_id,
                                    "locale": payload.language,
                                    "execution_mode": "demo",
                                    "demo_profile": demo_fixture.model_dump(mode="json"),
                                }
                            },
                        )
                        response_state = result.get("response", {})
                        interrupts = result.get("__interrupt__") or []
                        if interrupts:
                            first = interrupts[0]
                            value = first.value if hasattr(first, "value") else {}
                            message = str(value.get("question") or "Se necesita una respuesta.")
                        else:
                            message = str(response_state.get("user_message") or "")
                        response = DemoChatResponse(
                            message=message or "No fue posible completar la demo.",
                            profile_version=demo_fixture.profile_version,
                        )
                record_application_output(
                    span,
                    response.message,
                    {
                        "victus.demo.read_only": True,
                        "victus.demo.policy_denied": policy_denied,
                    },
                )
        except Exception:
            return JSONResponse({"error": "demo unavailable"}, status_code=503)
        return JSONResponse(response.model_dump(mode="json"))

    return Starlette(
        debug=False,
        lifespan=lifespan,
        routes=[
            Route("/health", health),
            Route("/chat", chat, methods=["POST"]),
            Route("/chat/debug", chat_debug, methods=["POST"]),
            Route("/demo/chat", demo_chat, methods=["POST"]),
        ],
    )


def _bearer_token(request: Request) -> str | None:
    authorization = request.headers.get("authorization", "")
    scheme, _, token = authorization.partition(" ")
    return token if scheme.lower() == "bearer" and token else None


def _environment_flag(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _debug_state(state: dict[str, Any]) -> dict[str, Any]:
    return {
        field: _debug_value(state[field])
        for field in DEBUG_STATE_FIELDS
        if field in state
    }


def _debug_value(value: Any, *, depth: int = 0) -> Any:
    if depth >= MAX_DEBUG_DEPTH:
        return "[truncated]"
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return value if len(value) <= MAX_DEBUG_STRING else f"{value[:MAX_DEBUG_STRING]}…"
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, BaseModel):
        return _debug_value(value.model_dump(mode="python"), depth=depth + 1)
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for index, (key, item) in enumerate(value.items()):
            if index >= MAX_DEBUG_ITEMS:
                result["__truncated__"] = True
                break
            field = str(key)
            if field in {"additional_kwargs", "response_metadata"} and item == {}:
                continue
            normalized = field.lower().replace("-", "_")
            result[field] = (
                "[redacted]"
                if any(part in normalized for part in SENSITIVE_FIELD_PARTS)
                else _debug_value(item, depth=depth + 1)
            )
        return result
    if isinstance(value, (list, tuple, set, frozenset)):
        items = list(value)
        serialized = [
            _debug_value(item, depth=depth + 1) for item in items[:MAX_DEBUG_ITEMS]
        ]
        if len(items) > MAX_DEBUG_ITEMS:
            serialized.append("[truncated]")
        return serialized
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return _debug_value(model_dump(), depth=depth + 1)
    return _debug_value(str(value), depth=depth + 1)


def _chat_response(conversation_id: str, state: dict[str, Any]) -> ChatResponse:
    interrupts = state.get("__interrupt__") or []
    if interrupts:
        item = interrupts[0]
        value = item.value if hasattr(item, "value") else item.get("value", {})
        interrupt_id = item.id if hasattr(item, "id") else str(item.get("id", ""))
        value = value if isinstance(value, dict) else {}
        interrupt = ChatInterrupt(
            id=interrupt_id,
            kind=str(value.get("kind") or "input"),
            question=str(value.get("question") or "Se necesita una respuesta."),
            details={key: val for key, val in value.items() if key not in {"kind", "question"}},
        )
        return ChatResponse(
            conversation_id=conversation_id,
            status="needs_user_response",
            message=interrupt.question,
            interrupt=interrupt,
        )
    response = state.get("response", {})
    tool = state.get("tool_context", {}).get("last_tool_result")
    mode = response.get("mode", "error")
    status = "blocked" if mode == "blocked" else "error" if mode == "error" else "completed"
    events = list((tool or {}).get("events_emitted", []))
    return ChatResponse(
        conversation_id=conversation_id,
        status=status,
        message=str(response.get("user_message") or ""),
        tool=tool,
        events=events,
    )


def main() -> None:
    import uvicorn

    uvicorn.run(
        create_app(),
        host=os.getenv("VICTUS_CHAT_HTTP_HOST", DEFAULT_HOST),
        port=int(os.getenv("VICTUS_CHAT_HTTP_PORT", str(DEFAULT_PORT))),
    )


if __name__ == "__main__":
    main()
