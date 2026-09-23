import asyncio
from contextlib import nullcontext

from domain.events.envelope import UserEventEnvelope
from ops.scripts.phoenix_intent_eval import (
    build_contract_evaluator,
    dataset_examples,
    evaluation_passed,
)
from ops.scripts.mcp_intent_eval import function_tools, score_case
from tools.catalog import get_tool, list_tools
from tools.contracts import ToolContext, ToolIdentity, ToolInvocation
from tools.runtime import ToolRuntime
from victus_platform.llm.contracts import LLMRequest, LLMResponse
from victus_platform.llm.litellm_client import LiteLLMClient
from victus_platform.telemetry.phoenix import (
    _current_trace_context,
    capture_phoenix_trace_context,
    initialize_phoenix,
    record_llm_response,
    trace_llm_call,
    _record_llm_request,
)


def test_catalog_and_capabilities_share_one_runtime() -> None:
    definitions = list_tools(exposure="mcp")
    names = [definition.name for definition in definitions]
    assert names == ["event_capture"]
    assert all(get_tool(name).input_schema for name in names)
    assert all("Use " in definition.description for definition in definitions)
    assert all("Do not use" in definition.description for definition in definitions)
    assert "meal or beverage" in get_tool("event_capture").description
    event_schema = get_tool("event_capture").input_schema
    item_def = event_schema["properties"]["items"]["items"]
    assert "$defs" not in event_schema
    assert "$ref" not in item_def
    assert set(event_schema["properties"]) == {"items", "occurred_at_text"}
    assert event_schema["required"] == ["items"]
    assert item_def["required"] == ["name", "quantity", "unit"]
    assert "name" in item_def["properties"]
    assert "quantity" in item_def["properties"]
    assert "unit" in item_def["properties"]
    assert item_def["properties"]["unit"]["anyOf"][0]["enum"] == ["g", "ml"]
    assert "food_label" not in item_def["properties"]

    store = FakeEventStore()
    result = ToolRuntime(lambda: nullcontext(store)).invoke(
        ToolInvocation(
            name="event_capture",
            arguments={"items": [{"name": "arroz", "quantity": 100, "unit": "g"}]},
            context=ToolContext(
                source="test", identity=ToolIdentity(subject="u1", authenticated=True)
            ),
        )
    )
    assert result.status == "success"
    assert result.events_emitted
    assert result.data["occurred_at_text"] == "today"


def test_runtime_applies_trace_idempotency_and_normalizes_errors() -> None:
    store = FakeEventStore()
    runtime = ToolRuntime(lambda: nullcontext(store))
    result = runtime.invoke(
        ToolInvocation(
            name="event_capture",
            arguments={
                "items": [{"name": "arroz", "quantity": 100, "unit": "g"}],
            },
            context=ToolContext(
                source="test",
                identity=ToolIdentity(subject="u1", authenticated=True),
                original_text="hoy comi arroz",
                trace_id="trace-1",
                idempotency_key="request-1",
            ),
        )
    )
    assert store.appended is not None
    assert store.appended.idempotency_key == "request-1"
    assert store.appended.metadata.trace_id == "trace-1"

    rejected = runtime.invoke(
        ToolInvocation(name="missing", arguments={}, context=ToolContext(source="test"))
    )
    assert rejected.status == "rejected"
    assert rejected.error and rejected.error.code == "invalid_invocation"


def test_event_capture_rejects_non_meal_actions() -> None:
    store = FakeEventStore()
    result = ToolRuntime(lambda: nullcontext(store)).invoke(
        ToolInvocation(
            name="event_capture",
            arguments={
                "capture_action": "log_symptom",
            },
            context=ToolContext(
                source="test", identity=ToolIdentity(subject="u1", authenticated=True)
            ),
        )
    )
    assert result.status == "rejected"
    assert result.events_emitted == []
    assert store.appended is None
    assert result.error is not None
    assert "Extra inputs" in result.error.message


def test_event_capture_records_the_supplied_meal_items() -> None:
    store = FakeEventStore()
    result = ToolRuntime(lambda: nullcontext(store)).invoke(
        ToolInvocation(
            name="event_capture",
            arguments={
                "items": [
                    {"name": "tallarines", "quantity": 150, "unit": "g"},
                    {"name": "salsa", "quantity": 30, "unit": "g"},
                ],
            },
            context=ToolContext(
                source="test", identity=ToolIdentity(subject="u1", authenticated=True)
            ),
        )
    )

    assert result.status == "success"
    assert result.events_emitted
    assert store.appended is not None


def test_event_capture_requests_clarification_for_missing_quantity() -> None:
    result = ToolRuntime(lambda: nullcontext(FakeEventStore())).invoke(
        ToolInvocation(
            name="event_capture",
            arguments={"items": [{"name": "arroz"}]},
            context=ToolContext(
                source="test", identity=ToolIdentity(subject="u1", authenticated=True)
            ),
        )
    )

    assert result.status == "needs_clarification"
    assert result.clarification is not None
    assert result.clarification.missing_fields == ["items[0].quantity", "items[0].unit"]


def test_event_capture_requests_clarification_for_inferred_single_gram() -> None:
    result = ToolRuntime(lambda: nullcontext(FakeEventStore())).invoke(
        ToolInvocation(
            name="event_capture",
            arguments={"items": [{"name": "pollo", "quantity": 1, "unit": "g"}]},
            context=ToolContext(
                source="test",
                identity=ToolIdentity(subject="u1", authenticated=True),
                original_text="hoy me comi un pollo",
            ),
        )
    )

    assert result.status == "needs_clarification"
    assert result.clarification is not None
    assert result.clarification.missing_fields == ["items[0].quantity", "items[0].unit"]
    assert result.events_emitted == []


def test_event_capture_rejects_noncanonical_item_shape() -> None:
    result = ToolRuntime(lambda: nullcontext(FakeEventStore())).invoke(
        ToolInvocation(
            name="event_capture",
            arguments={
                "items": [{"food_label": "fideos con salsa", "quantity": 100, "unit": "g"}],
            },
            context=ToolContext(
                source="test", identity=ToolIdentity(subject="u1", authenticated=True)
            ),
        )
    )

    assert result.status == "rejected"
    assert result.error is not None
    assert result.error.code == "invalid_invocation"
    assert "Extra inputs are not permitted" in result.error.message


def test_intent_eval_uses_catalog_and_checks_exact_input() -> None:
    tools = function_tools()
    assert [item["function"]["name"] for item in tools] == [
        definition.name for definition in list_tools(exposure="mcp")
    ]
    case = {
        "id": "meal",
        "input": "Hoy comi arroz",
        "expected_tool": "event_capture",
        "exact_input_argument": "normalized_text",
    }
    passed = score_case(
        case,
        [
            {
                "name": "event_capture",
                "arguments": {
                    "user_id": "mcp-smoke-user",
                    "normalized_text": "Hoy comi arroz",
                },
            }
        ],
        user_id="mcp-smoke-user",
    )
    changed = score_case(
        case,
        [
            {
                "name": "event_capture",
                "arguments": {
                    "user_id": "mcp-smoke-user",
                    "normalized_text": "El usuario comio arroz",
                },
            }
        ],
        user_id="mcp-smoke-user",
    )
    assert passed["passed"] is True
    assert changed["passed"] is False


def test_phoenix_intent_dataset_and_contract_evaluator_reuse_existing_scoring() -> None:
    cases = [
        {
            "id": "meal",
            "input": "Hoy comi arroz",
            "expected_tool": "event_capture",
            "exact_input_argument": "normalized_text",
        }
    ]
    examples = dataset_examples(cases)
    assert examples == [
        {
            "id": "meal",
            "input": {"message": "Hoy comi arroz"},
            "output": {
                "expected_tool": "event_capture",
                "expected_arguments": {},
                "exact_input_argument": "normalized_text",
            },
            "metadata": {"case_id": "meal"},
        }
    ]

    evaluator = build_contract_evaluator(user_id="mcp-smoke-user")
    result = evaluator(
        examples[0]["input"],
        {
            "tool_calls": [
                {
                    "name": "event_capture",
                    "arguments": {
                        "user_id": "mcp-smoke-user",
                        "normalized_text": "Hoy comi arroz",
                    },
                }
            ]
        },
        examples[0]["output"],
        examples[0]["metadata"],
    )
    assert result["score"] == 1
    assert evaluation_passed({"error": None, "result": result}) is True
    assert evaluation_passed({"error": "provider failed", "result": None}) is False


def test_phoenix_tracing_is_disabled_by_default(monkeypatch) -> None:
    monkeypatch.delenv("PHOENIX_TRACING_ENABLED", raising=False)
    assert initialize_phoenix() is None
    assert capture_phoenix_trace_context() is None
    with trace_llm_call(object()) as span:
        assert span is None


def test_llm_context_prefers_active_application_span() -> None:
    application_span = RecordingSpan(recording=True)
    framework_span = RecordingSpan(recording=True)
    trace = FakeTrace(application_span)
    context = FakeContext()

    resolved = _current_trace_context(trace, context, lambda: framework_span)

    assert resolved == ("span-context", application_span)
    assert trace.context_spans == [application_span]


def test_llm_context_falls_back_to_framework_span_when_no_application_span_is_active() -> None:
    framework_span = RecordingSpan(recording=True)
    trace = FakeTrace(RecordingSpan(recording=False))
    context = FakeContext()

    resolved = _current_trace_context(trace, context, lambda: framework_span)

    assert resolved == ("span-context", framework_span)


class RecordingSpan:
    def __init__(self, *, recording: bool) -> None:
        self.recording = recording

    def is_recording(self) -> bool:
        return self.recording


class FakeTrace:
    def __init__(self, current_span) -> None:
        self.current_span = current_span
        self.context_spans = []

    def get_current_span(self):
        return self.current_span

    def set_span_in_context(self, span):
        self.context_spans.append(span)
        return ("span-context", span)


class FakeContext:
    def get_current(self):
        return "ambient-context"


def test_phoenix_llm_attributes_expose_messages_tools_and_tool_calls() -> None:
    class Span:
        def __init__(self) -> None:
            self.attributes = {}

        def set_attribute(self, key, value) -> None:
            self.attributes[key] = value

    class Attributes:
        INPUT_VALUE = "input.value"
        INPUT_MIME_TYPE = "input.mime_type"
        LLM_INPUT_MESSAGES = "llm.input_messages"
        LLM_TOOLS = "llm.tools"
        LLM_INVOCATION_PARAMETERS = "llm.invocation_parameters"

    request = LLMRequest(
        operation="agent.decision",
        model="test-model",
        messages=[
            {"role": "system", "content": "Decide."},
            {"role": "user", "content": "Registra arroz."},
        ],
        temperature=0,
        tool_choice="auto",
        tools=[{"type": "function", "function": {"name": "event_capture"}}],
    )
    span = Span()

    _record_llm_request(span, request, Attributes)
    record_llm_response(
        span,
        {
            "choices": [
                {
                    "finish_reason": "tool_calls",
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call-1",
                                "function": {
                                    "name": "event_capture",
                                    "arguments": '{"capture_action":"log_meal"}',
                                },
                            }
                        ],
                    },
                }
            ]
        },
    )

    assert span.attributes["llm.input_messages.0.message.content"] == "Decide."
    assert span.attributes["llm.input_messages.1.message.content"] == "Registra arroz."
    assert span.attributes["llm.tools.0.tool.json_schema"] == (
        '{"type":"function","function":{"name":"event_capture"}}'
    )
    assert span.attributes["llm.invocation_parameters"] == '{"temperature":0,"tool_choice":"auto"}'
    assert span.attributes["llm.output_messages.0.message.tool_calls.0.tool_call.function.name"] == (
        "event_capture"
    )
    assert span.attributes[
        "llm.output_messages.0.message.tool_calls.0.tool_call.function.arguments"
    ] == '{"capture_action":"log_meal"}'


def test_async_litellm_call_forwards_active_phoenix_context(monkeypatch) -> None:
    from victus_platform.llm import litellm_client

    expected_parent_context = object()
    request = LLMRequest(operation="agent.decision", model="test", messages=[])
    expected = LLMResponse(text="ok")
    client = LiteLLMClient()

    monkeypatch.setattr(
        litellm_client,
        "capture_phoenix_trace_context",
        lambda: expected_parent_context,
    )

    def complete(request_arg, *, parent_context=None):
        assert request_arg is request
        assert parent_context is expected_parent_context
        return expected

    monkeypatch.setattr(client, "complete", complete)

    assert asyncio.run(client.acomplete(request)) is expected


class FakeEventStore:
    appended: UserEventEnvelope | None = None

    def append(self, event: UserEventEnvelope) -> UserEventEnvelope:
        self.appended = event.model_copy(update={"event_seq": 1})
        return self.appended
