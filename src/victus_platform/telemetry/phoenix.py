from __future__ import annotations

import json
import os
from contextlib import contextmanager, nullcontext
from collections.abc import Mapping
from typing import Any, Iterator


def initialize_phoenix() -> Any | None:
    if not _environment_flag("PHOENIX_TRACING_ENABLED"):
        return None
    try:
        from phoenix.otel import register
    except ImportError as exc:
        raise RuntimeError(
            "Phoenix tracing is enabled but the 'phoenix' optional dependencies are not installed"
        ) from exc

    return register(auto_instrument=False, batch=True, verbose=False)


def shutdown_phoenix(provider: Any | None) -> None:
    if provider is not None:
        provider.shutdown()


def phoenix_context(
    *,
    session_id: str,
    user_id: str,
    metadata: dict[str, Any] | None = None,
):
    if not _environment_flag("PHOENIX_TRACING_ENABLED"):
        return nullcontext()
    try:
        from phoenix.otel import using_attributes
    except ImportError as exc:
        raise RuntimeError(
            "Phoenix tracing is enabled but the 'phoenix' optional dependencies are not installed"
        ) from exc
    return using_attributes(session_id=session_id, user_id=user_id, metadata=metadata)


def capture_phoenix_trace_context() -> Any | None:
    if not _environment_flag("PHOENIX_TRACING_ENABLED"):
        return None
    try:
        from openinference.instrumentation.langchain import get_current_span
        from opentelemetry import context, trace
    except ImportError as exc:
        raise RuntimeError(
            "Phoenix tracing is enabled but the 'phoenix' optional dependencies are not installed"
        ) from exc
    return _current_trace_context(trace, context, get_current_span)


def _current_trace_context(trace: Any, context: Any, get_langchain_span: Any) -> Any:
    """Prefer the active application span over framework instrumentation context."""
    active_span = trace.get_current_span()
    if active_span and active_span.is_recording():
        return trace.set_span_in_context(active_span)
    if parent_span := get_langchain_span():
        return trace.set_span_in_context(parent_span)
    return context.get_current()


@contextmanager
def trace_chat_request(
    *,
    headers: Mapping[str, str],
    attributes: Mapping[str, str | int | bool],
) -> Iterator[Any | None]:
    """Continue a gateway trace at the authenticated agent HTTP boundary."""
    if not _environment_flag("PHOENIX_TRACING_ENABLED"):
        yield None
        return
    try:
        from opentelemetry import propagate, trace
        from opentelemetry.trace import Status, StatusCode
    except ImportError as exc:
        raise RuntimeError(
            "Phoenix tracing is enabled but the 'phoenix' optional dependencies are not installed"
        ) from exc

    parent_context = propagate.extract(dict(headers))
    tracer = trace.get_tracer("victus-agent-chat")
    with tracer.start_as_current_span("agent.http.chat", context=parent_context) as span:
        set_current_span_attributes(
            {
                "service.name": "victus-agent-chat",
                "victus.component": "agent",
                **attributes,
            }
        )
        try:
            yield span
        except Exception as exc:
            span.record_exception(exc)
            span.set_status(Status(StatusCode.ERROR, str(exc)))
            raise
        else:
            span.set_status(Status(StatusCode.OK))


def set_current_span_attributes(attributes: Mapping[str, Any]) -> None:
    """Add concise, non-secret attributes to the current Phoenix/OpenTelemetry span."""
    if not _environment_flag("PHOENIX_TRACING_ENABLED"):
        return
    try:
        from opentelemetry import trace
    except ImportError as exc:
        raise RuntimeError(
            "Phoenix tracing is enabled but the 'phoenix' optional dependencies are not installed"
        ) from exc
    span = trace.get_current_span()
    if not span.is_recording():
        return
    for key, value in attributes.items():
        if _safe_attribute(key, value):
            span.set_attribute(key, value)


@contextmanager
def trace_application_span(
    name: str,
    *,
    input_value: str,
    attributes: Mapping[str, Any] | None = None,
    span_kind: str = "CHAIN",
) -> Iterator[Any | None]:
    """Record a concise application operation while preserving raw child spans separately."""
    if not _environment_flag("PHOENIX_TRACING_ENABLED"):
        yield None
        return
    try:
        from opentelemetry import trace
        from opentelemetry.trace import Status, StatusCode
        from phoenix.otel import OpenInferenceSpanKindValues, SpanAttributes
    except ImportError as exc:
        raise RuntimeError(
            "Phoenix tracing is enabled but the 'phoenix' optional dependencies are not installed"
        ) from exc

    tracer = trace.get_tracer("victus-agent.application")
    kind = getattr(OpenInferenceSpanKindValues, span_kind).value
    with tracer.start_as_current_span(name) as span:
        span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, kind)
        span.set_attribute(SpanAttributes.INPUT_VALUE, input_value)
        span.set_attribute(SpanAttributes.INPUT_MIME_TYPE, "text/plain")
        set_current_span_attributes({"victus.layer": "application", **(attributes or {})})
        try:
            yield span
        except Exception as exc:
            span.record_exception(exc)
            span.set_status(Status(StatusCode.ERROR, str(exc)))
            raise
        else:
            span.set_status(Status(StatusCode.OK))


def record_application_output(
    span: Any | None,
    output_value: str,
    attributes: Mapping[str, Any] | None = None,
) -> None:
    """Attach human-readable outcome attributes to an application span."""
    if span is None:
        return
    from phoenix.otel import SpanAttributes

    span.set_attribute(SpanAttributes.OUTPUT_VALUE, output_value)
    span.set_attribute(SpanAttributes.OUTPUT_MIME_TYPE, "text/plain")
    set_current_span_attributes(attributes or {})


@contextmanager
def trace_llm_call(request: Any, *, parent_context: Any | None = None) -> Iterator[Any | None]:
    if not _environment_flag("PHOENIX_TRACING_ENABLED"):
        yield None
        return
    try:
        from opentelemetry import trace
        from phoenix.otel import OpenInferenceSpanKindValues, SpanAttributes
    except ImportError as exc:
        raise RuntimeError(
            "Phoenix tracing is enabled but the 'phoenix' optional dependencies are not installed"
        ) from exc

    tracer = trace.get_tracer("victus-agent.llm")
    operation = str(request.operation).replace(".", "_")
    with tracer.start_as_current_span(f"llm.{operation}", context=parent_context) as span:
        span.set_attribute(
            SpanAttributes.OPENINFERENCE_SPAN_KIND,
            OpenInferenceSpanKindValues.LLM.value,
        )
        span.set_attribute(SpanAttributes.LLM_MODEL_NAME, str(request.model))
        span.set_attribute("victus.llm.operation", str(request.operation))
        if request.redact_content:
            span.set_attribute(SpanAttributes.INPUT_VALUE, "[redacted]")
            span.set_attribute(SpanAttributes.INPUT_MIME_TYPE, "text/plain")
            span.set_attribute("victus.llm.content_redacted", True)
        else:
            _record_llm_request(span, request, SpanAttributes)
        for key, value in request.metadata.items():
            if _safe_attribute(key, value):
                span.set_attribute(f"victus.metadata.{key}", value)
        yield span


def record_llm_response(
    span: Any | None,
    raw_response: Mapping[str, Any],
    *,
    redact_content: bool = False,
) -> None:
    """Record the provider response using Phoenix's OpenInference LLM attributes."""
    if span is None:
        return
    from phoenix.otel import SpanAttributes

    choices = raw_response.get("choices")
    choice = choices[0] if isinstance(choices, list) and choices else None
    message = choice.get("message") if isinstance(choice, Mapping) else None
    if not isinstance(message, Mapping):
        return

    if redact_content:
        span.set_attribute(SpanAttributes.OUTPUT_VALUE, "[redacted]")
        span.set_attribute(SpanAttributes.OUTPUT_MIME_TYPE, "text/plain")
        return

    normalized = _normalized_message(message)
    span.set_attribute(SpanAttributes.OUTPUT_VALUE, _json_value(message))
    span.set_attribute(SpanAttributes.OUTPUT_MIME_TYPE, "application/json")
    _record_message(span, SpanAttributes.LLM_OUTPUT_MESSAGES, 0, normalized)
    if isinstance(choice.get("finish_reason"), str):
        span.set_attribute(SpanAttributes.LLM_FINISH_REASON, choice["finish_reason"])


def record_llm_usage(span: Any | None, usage: dict[str, Any]) -> None:
    if span is None:
        return
    from phoenix.otel import SpanAttributes

    counts = (
        ("prompt_tokens", SpanAttributes.LLM_TOKEN_COUNT_PROMPT),
        ("completion_tokens", SpanAttributes.LLM_TOKEN_COUNT_COMPLETION),
        ("total_tokens", SpanAttributes.LLM_TOKEN_COUNT_TOTAL),
    )
    for source, attribute in counts:
        value = usage.get(source)
        if isinstance(value, int):
            span.set_attribute(attribute, value)


def _safe_attribute(key: str, value: Any) -> bool:
    normalized = str(key).lower().replace("-", "_")
    if any(part in normalized for part in ("api_key", "authorization", "password", "secret", "token")):
        return False
    return isinstance(value, (bool, int, float, str))


def _record_llm_request(span: Any, request: Any, attributes: Any) -> None:
    messages = [_normalized_message(message) for message in request.messages]
    span.set_attribute(attributes.INPUT_VALUE, _json_value(request.messages))
    span.set_attribute(attributes.INPUT_MIME_TYPE, "application/json")
    for index, message in enumerate(messages):
        _record_message(span, attributes.LLM_INPUT_MESSAGES, index, message)

    for index, tool in enumerate(request.tools or []):
        span.set_attribute(
            f"{attributes.LLM_TOOLS}.{index}.tool.json_schema",
            _json_value(tool),
        )

    parameters = {
        key: value
        for key, value in {
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "response_format": request.response_format,
            "tool_choice": request.tool_choice,
        }.items()
        if value is not None
    }
    if parameters:
        span.set_attribute(attributes.LLM_INVOCATION_PARAMETERS, _json_value(parameters))


def _record_message(span: Any, prefix: str, index: int, message: Mapping[str, Any]) -> None:
    base = f"{prefix}.{index}.message"
    for field in ("role", "content", "name", "tool_call_id"):
        value = message.get(field)
        if value is not None:
            span.set_attribute(f"{base}.{field}", value)

    tool_calls = message.get("tool_calls")
    if not isinstance(tool_calls, list):
        return
    for tool_index, tool_call in enumerate(tool_calls):
        if not isinstance(tool_call, Mapping):
            continue
        call_base = f"{base}.tool_calls.{tool_index}.tool_call"
        if tool_call.get("id") is not None:
            span.set_attribute(f"{call_base}.id", str(tool_call["id"]))
        function = tool_call.get("function")
        if not isinstance(function, Mapping):
            continue
        if function.get("name") is not None:
            span.set_attribute(f"{call_base}.function.name", str(function["name"]))
        if function.get("arguments") is not None:
            span.set_attribute(
                f"{call_base}.function.arguments",
                _tool_call_arguments(function["arguments"]),
            )


def _normalized_message(message: Mapping[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for field in ("role", "content", "name", "tool_call_id", "tool_calls"):
        value = message.get(field)
        if value is None:
            continue
        normalized[field] = value if field != "content" or isinstance(value, str) else _json_value(value)
    return normalized


def _json_value(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), default=str)


def _tool_call_arguments(value: Any) -> str:
    return value if isinstance(value, str) else _json_value(value)


def _environment_flag(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}
