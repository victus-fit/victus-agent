from __future__ import annotations

import re

from adapters.langgraph.engine.state import VictusGraphState
from adapters.langgraph.prompts.compose_response import (
    COMPOSE_RESPONSE_SYSTEM_PROMPT,
    compose_response_user_prompt,
)
from victus_platform.llm.contracts import LLMClient, LLMRequest
from domain.shared.text import normalize_text
from tools.catalog import list_tools
from victus_platform.safety.rules import SafetyPrecheck, SafetyPrecheckInput

AGENT_ENABLED_TOOLS = frozenset({"event_capture", "evidence_retrieval"})


def normalize_request(state: VictusGraphState) -> VictusGraphState:
    request = dict(state.get("request", {}))
    original_text = str(request.get("original_text") or request.get("raw_text", ""))
    existing_working_text = str(request.get("working_text") or "")
    normalized_text = existing_working_text or normalize_text(original_text)
    request = _clean_request(request)
    request["original_text"] = original_text
    request["working_text"] = normalized_text
    return _merge(
        state,
        request=request,
        node_name="normalize_request",
        transform={"step": "normalize", "input": original_text, "output": normalized_text},
    )


def safety_precheck(*, llm_client: LLMClient | None = None, model: str | None = None):
    def node(state: VictusGraphState) -> VictusGraphState:
        request = state.get("request", {})
        text = str(
            request.get("working_text")
            or request.get("original_text")
            or request.get("raw_text")
            or ""
        )
        if not llm_client or not model:
            local = SafetyPrecheck().check(
                SafetyPrecheckInput(
                    original_text=str(request.get("original_text") or text),
                    working_text=text,
                )
            )
            if local.decision != "allow":
                return _merge(
                    state,
                    safety={
                        "status": "blocked",
                        "reasons": local.reasons,
                        "decision": local.decision,
                        "severity": local.severity,
                        "categories": local.categories,
                    },
                    node_name="safety_precheck",
                )
            return _merge(state, safety=_allowed_safety(), node_name="safety_precheck")

        result = llm_client.complete(
            LLMRequest(
                operation="agent.safety_precheck.llama_guard",
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": _llama_guard_prompt(text),
                    }
                ],
                temperature=0,
                max_tokens=32,
                metadata={"guard": "llama_guard_3_1b"},
            )
        )
        safety = _safety_from_llama_guard(result.text)
        return _merge(state, safety=safety, node_name="safety_precheck")

    return node


def _llama_guard_prompt(user_text: str) -> str:
    return (
        "Task: Check if the user message is safe or unsafe. "
        "Respond only with 'safe' or 'unsafe' followed by the violated category if any.\n\n"
        f"User message:\n{user_text}"
    )


def _safety_from_llama_guard(output: str) -> dict[str, object]:
    normalized = output.strip().lower()
    if normalized.startswith("unsafe"):
        categories = _llama_guard_categories(normalized)
        reasons = [f"llama_guard_{category}" for category in categories]
        return {
            "status": "blocked",
            "reasons": reasons,
            "decision": "route_to_safety_triage",
            "severity": "high",
            "categories": categories,
        }
    return _allowed_safety()


def _llama_guard_categories(normalized_output: str) -> list[str]:
    categories_by_code = {
        "s1": "violent_crimes",
        "s2": "non_violent_crimes",
        "s3": "sex_related_crimes",
        "s4": "child_sexual_exploitation",
        "s5": "defamation",
        "s6": "specialized_advice",
        "s7": "privacy",
        "s8": "intellectual_property",
        "s9": "indiscriminate_weapons",
        "s10": "hate",
        "s11": "self_harm",
        "s12": "sexual_content",
        "s13": "elections",
        "s14": "code_interpreter_abuse",
    }
    categories = []
    for code, category in categories_by_code.items():
        if re.search(rf"(?<![a-z0-9]){code}(?![a-z0-9])", normalized_output):
            categories.append(category)
    return categories or ["unsafe"]


def _safety_from_prompt_guard(label: str) -> dict[str, object]:
    normalized = label.strip().upper()
    if normalized == "BENIGN":
        return _allowed_safety()

    if normalized == "INJECTION":
        category = "prompt_injection"
    elif normalized == "JAILBREAK":
        category = "jailbreak"
    else:
        category = "prompt_guard_unknown"
    reason_code = f"prompt_guard_{category}"
    return {
        "status": "blocked",
        "reasons": [reason_code],
        "decision": "route_to_safety_triage",
        "severity": "high",
        "categories": [category],
    }


def _allowed_safety() -> dict[str, object]:
    return {
        "status": "ok",
        "reasons": [],
        "decision": "allow",
        "severity": "none",
        "categories": ["none"],
    }


def tool_registry(*, enabled_tools: frozenset[str] = AGENT_ENABLED_TOOLS):
    def node(state: VictusGraphState) -> VictusGraphState:
        safety = state.get("safety", {})
        if safety.get("status") == "blocked":
            allowed_tools: list[str] = []
        else:
            allowed_tools = [
                tool.name
                for tool in list_tools(exposure="langgraph")
                if tool.name in enabled_tools
            ]

        tool_context = dict(state.get("tool_context", {}))
        tool_context["allowed_tools"] = allowed_tools
        tool_context.setdefault("tool_results", [])
        return _merge(state, tool_context=tool_context, node_name="tool_registry")

    return node


def safety_blocked_response(state: VictusGraphState) -> VictusGraphState:
    safety = state.get("safety", {})
    tool_context = dict(state.get("tool_context", {}))
    tool_context["allowed_tools"] = []
    tool_context.setdefault("tool_results", [])
    return _merge(
        state,
        tool_context=tool_context,
        response=_blocked_response(list(safety.get("reasons", []))),
        node_name="safety_blocked_response",
    )


def compose_response(*, llm_client: LLMClient | None = None, model: str | None = None):
    if not llm_client or not model:
        return _compose_response_sync

    async def node(state: VictusGraphState) -> VictusGraphState:
        existing_response = state.get("response", {})
        if existing_response.get("user_message"):
            return _merge(state, node_name="compose_response")

        safety = state.get("safety", {})
        intent = state.get("intent", {})
        if safety.get("status") == "blocked":
            return _merge(
                state,
                response=_blocked_response(list(safety.get("reasons", []))),
                node_name="compose_response",
            )

        request = state.get("request", {})
        llm_response = await llm_client.acomplete(
            LLMRequest(
                operation="agent.compose_response",
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": COMPOSE_RESPONSE_SYSTEM_PROMPT,
                    },
                    {
                        "role": "user",
                        "content": compose_response_user_prompt(
                            route=str(intent.get("target_node", "unknown")),
                            raw_text=str(request.get("original_text", "")),
                        ),
                    },
                ],
                temperature=0,
                max_tokens=300,
                metadata={"target_node": intent.get("target_node", "unknown")},
            )
        )
        response = {
            "mode": "final",
            "user_message": llm_response.text,
        }
        return _merge(state, response=response, node_name="compose_response")


    return node


def _compose_response_sync(state: VictusGraphState) -> VictusGraphState:
    existing_response = state.get("response", {})
    if existing_response.get("user_message"):
        return _merge(state, node_name="compose_response")

    safety = state.get("safety", {})
    intent = state.get("intent", {})
    if safety.get("status") == "blocked":
        return _merge(
            state,
            response=_blocked_response(list(safety.get("reasons", []))),
            node_name="compose_response",
        )

    response = {
        "mode": "final",
        "user_message": f"Route selected: {intent.get('target_node', 'unknown')}",
    }
    return _merge(state, response=response, node_name="compose_response")


def _blocked_response(reasons: list[str]) -> dict[str, object]:
    return {
        "mode": "blocked",
        "user_message": (
            "No puedo ayudar con esa accion de forma segura. Busca apoyo medico urgente "
            "si los sintomas son graves o inmediatos."
        ),
    }


def _clean_request(request: dict[str, object]) -> dict[str, object]:
    for key in ("raw_text", "normalized_text", "english_safety_text", "tool_query"):
        request.pop(key, None)
    return request


def _merge(
    state: VictusGraphState,
    *,
    node_name: str,
    transform: dict[str, str] | None = None,
    **updates: object,
) -> VictusGraphState:
    audit = dict(state.get("audit", {}))
    audit.setdefault("node_path", [])
    audit.setdefault("events_emitted", [])
    audit.setdefault("warnings", [])
    audit.setdefault("errors", [])
    audit.setdefault("transforms", [])
    audit["node_path"] = [*audit["node_path"], node_name]
    if transform:
        audit["transforms"] = [*audit["transforms"], transform]
    return {**state, **updates, "audit": audit}
