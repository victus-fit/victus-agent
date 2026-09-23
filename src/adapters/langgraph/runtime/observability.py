from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from adapters.langgraph.engine.state import VictusGraphState
from victus_platform.telemetry.phoenix import record_application_output, trace_application_span


Summary = tuple[str, dict[str, str | int | float | bool]]
SyncNode = Callable[[VictusGraphState], VictusGraphState]
AsyncNode = Callable[[VictusGraphState], Awaitable[VictusGraphState]]


def observe_sync_node(
    name: str,
    node: SyncNode,
    *,
    input_summary: Callable[[VictusGraphState], Summary],
    output_summary: Callable[[VictusGraphState], Summary],
    span_kind: str = "CHAIN",
) -> SyncNode:
    def observed(state: VictusGraphState) -> VictusGraphState:
        input_value, input_attributes = input_summary(state)
        with trace_application_span(
            name,
            input_value=input_value,
            attributes=input_attributes,
            span_kind=span_kind,
        ) as span:
            result = node(state)
            output_value, output_attributes = output_summary(result)
            record_application_output(span, output_value, output_attributes)
            return result

    return observed


def observe_async_node(
    name: str | Callable[[VictusGraphState], str],
    node: AsyncNode,
    *,
    input_summary: Callable[[VictusGraphState], Summary],
    output_summary: Callable[[VictusGraphState], Summary],
    span_kind: str = "CHAIN",
) -> AsyncNode:
    async def observed(state: VictusGraphState) -> VictusGraphState:
        input_value, input_attributes = input_summary(state)
        operation_name = name(state) if callable(name) else name
        with trace_application_span(
            operation_name,
            input_value=input_value,
            attributes=input_attributes,
            span_kind=span_kind,
        ) as span:
            result = await node(state)
            output_value, output_attributes = output_summary(result)
            record_application_output(span, output_value, output_attributes)
            return result

    return observed


def request_summary(state: VictusGraphState) -> Summary:
    request = state.get("request", {})
    text = str(request.get("original_text") or request.get("raw_text") or "")
    return text, {"victus.request_id": str(request.get("request_id") or "")}


def safety_summary(state: VictusGraphState) -> Summary:
    safety = state.get("safety", {})
    categories = [str(item) for item in safety.get("categories", [])]
    return (
        f"Safety {safety.get('status', 'unknown')}: {', '.join(categories) or 'no categories'}",
        {
            "victus.safety.status": str(safety.get("status") or "unknown"),
            "victus.safety.decision": str(safety.get("decision") or ""),
            "victus.safety.severity": str(safety.get("severity") or ""),
            "victus.safety.categories": ",".join(categories),
        },
    )


def decision_summary(state: VictusGraphState) -> Summary:
    proposal = _proposal(state)
    response = state.get("response", {})
    if proposal.get("tool_name"):
        tool_name = str(proposal["tool_name"])
        return f"Selected tool: {tool_name}", {"victus.decision": "tool", "victus.tool.name": tool_name}
    if response.get("user_message"):
        return "Selected direct response", {"victus.decision": "final"}
    return "Awaiting model decision", {"victus.decision": "pending"}


def tool_input_summary(state: VictusGraphState) -> Summary:
    proposal = _proposal(state)
    name = str(proposal.get("tool_name") or "unknown")
    arguments = proposal.get("arguments") if isinstance(proposal.get("arguments"), dict) else {}
    if name == "event_capture":
        items = arguments.get("items") if isinstance(arguments.get("items"), list) else []
        details = "; ".join(_meal_item(item) for item in items if isinstance(item, dict))
        return details or "Meal capture requested", {"victus.tool.name": name, "victus.tool.item_count": len(items)}
    if name == "evidence_retrieval":
        query = str(arguments.get("query") or "")
        return query or "Evidence retrieval requested", {"victus.tool.name": name}
    return f"{name} requested", {"victus.tool.name": name}


def tool_output_summary(state: VictusGraphState) -> Summary:
    result = _last_tool_result(state)
    name = str(result.get("tool_name") or _proposal(state).get("tool_name") or "unknown")
    status = str(result.get("status") or "unknown")
    data = result.get("data") if isinstance(result.get("data"), dict) else {}
    attributes: dict[str, str | int | float | bool] = {
        "victus.tool.name": name,
        "victus.tool.status": status,
        "victus.tool.events_emitted": len(result.get("events_emitted") or []),
    }
    if name == "event_capture" and status == "success":
        items = data.get("items") if isinstance(data.get("items"), list) else []
        attributes["victus.tool.item_count"] = len(items)
        return (
            f"Meal captured: {'; '.join(_meal_item(item) for item in items if isinstance(item, dict))}",
            attributes,
        )
    if name == "evidence_retrieval" and status == "success":
        results = data.get("results") if isinstance(data.get("results"), list) else []
        citations = [
            str(item.get("canonical_evidence_id"))
            for item in results
            if isinstance(item, dict) and item.get("canonical_evidence_id")
        ]
        attributes["victus.retrieval.result_count"] = len(results)
        return f"Evidence retrieved: {len(results)} result(s) [{', '.join(citations)}]", attributes
    error = result.get("error") if isinstance(result.get("error"), dict) else {}
    return f"{name}: {status}{': ' + str(error.get('message')) if error.get('message') else ''}", attributes


def response_summary(state: VictusGraphState) -> Summary:
    response = state.get("response", {})
    message = str(response.get("user_message") or "")
    return message or "Response completed", {"victus.response.mode": str(response.get("mode") or "unknown")}


def _proposal(state: VictusGraphState) -> dict[str, Any]:
    context = state.get("tool_context", {})
    proposal = context.get("proposed_action") if isinstance(context, dict) else {}
    return proposal if isinstance(proposal, dict) else {}


def _last_tool_result(state: VictusGraphState) -> dict[str, Any]:
    context = state.get("tool_context", {})
    result = context.get("last_tool_result") if isinstance(context, dict) else {}
    return result if isinstance(result, dict) else {}


def _meal_item(item: dict[str, Any]) -> str:
    name = str(item.get("name") or "unknown item")
    quantity = item.get("quantity")
    unit = item.get("unit")
    return f"{name} {quantity} {unit}" if quantity is not None and unit else name
