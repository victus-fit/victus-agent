from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from adapters.langgraph.engine.agent import (
    agent_decision,
    clarification_interrupt,
    compose_final_response,
    confirmation_interrupt,
    execute_tool,
    finalize_turn,
    ingest_turn,
    route_after_clarification,
    route_after_confirmation,
    route_after_decision,
    route_after_execution,
)
from adapters.langgraph.runtime.context import (
    AGENT_ENABLED_TOOLS,
    normalize_request,
    safety_blocked_response,
    safety_precheck,
    tool_registry,
)
from adapters.langgraph.runtime.observability import (
    decision_summary,
    observe_async_node,
    observe_sync_node,
    request_summary,
    response_summary,
    safety_summary,
    tool_input_summary,
    tool_output_summary,
)
from adapters.langgraph.engine.routing import route_after_safety
from adapters.langgraph.engine.state import VictusGraphState
from adapters.langgraph.runtime.memory import recall_long_term_memory, update_long_term_memory
from bootstrap.runtime import build_runtime
from victus_platform.config.runtime import load_runtime_config
from victus_platform.llm.contracts import LLMClient
from victus_platform.llm.factory import build_llm_client


def build_graph(
    *,
    llm_client: LLMClient | None = None,
    safety_client: LLMClient | None = None,
    tool_runtime=None,
    checkpointer=None,
    store=None,
    enabled_tools: frozenset[str] | None = None,
    redact_llm_content: bool = False,
):
    runtime_config = load_runtime_config()
    tool_runtime = tool_runtime or build_runtime()
    graph_builder = StateGraph(VictusGraphState)
    graph_builder.add_node("ingest_turn", ingest_turn)
    graph_builder.add_node("normalize_request", normalize_request)
    graph_builder.add_node("recall_long_term_memory", recall_long_term_memory(store))
    graph_builder.add_node(
        "safety_precheck",
        observe_sync_node(
            "agent.safety_precheck",
            safety_precheck(
                llm_client=safety_client,
                model=runtime_config.safety.model if safety_client else None,
            ),
            input_summary=request_summary,
            output_summary=safety_summary,
        ),
    )
    graph_builder.add_node("safety_blocked_response", safety_blocked_response)
    graph_builder.add_node(
        "tool_registry", tool_registry(enabled_tools=enabled_tools or AGENT_ENABLED_TOOLS)
    )
    graph_builder.add_node(
        "agent_decision",
        observe_async_node(
            "agent.decision",
            agent_decision(
                llm_client=llm_client,
                model=runtime_config.llm.model,
                redact_content=redact_llm_content,
            ),
            input_summary=request_summary,
            output_summary=decision_summary,
        ),
    )
    graph_builder.add_node("confirmation_interrupt", confirmation_interrupt)
    graph_builder.add_node(
        "execute_tool",
        observe_async_node(
            lambda state: f"tool.{tool_input_summary(state)[1]['victus.tool.name']}",
            execute_tool(tool_runtime),
            input_summary=tool_input_summary,
            output_summary=tool_output_summary,
            span_kind="TOOL",
        ),
    )
    graph_builder.add_node(
        "clarification_interrupt",
        clarification_interrupt(
            llm_client=llm_client,
            model=runtime_config.llm.model,
            redact_content=redact_llm_content,
        ),
    )
    graph_builder.add_node(
        "compose_final_response",
        observe_sync_node(
            "agent.final_response",
            compose_final_response,
            input_summary=decision_summary,
            output_summary=response_summary,
        ),
    )
    graph_builder.add_node("update_long_term_memory", update_long_term_memory(store))
    graph_builder.add_node("finalize_turn", finalize_turn)

    graph_builder.add_edge(START, "ingest_turn")
    graph_builder.add_edge("ingest_turn", "normalize_request")
    graph_builder.add_edge("normalize_request", "recall_long_term_memory")
    graph_builder.add_edge("recall_long_term_memory", "safety_precheck")
    graph_builder.add_conditional_edges(
        "safety_precheck",
        route_after_safety,
        {"blocked": "safety_blocked_response", "allowed": "tool_registry"},
    )
    graph_builder.add_edge("safety_blocked_response", "update_long_term_memory")
    graph_builder.add_edge("tool_registry", "agent_decision")
    graph_builder.add_conditional_edges(
        "agent_decision",
        route_after_decision,
        {
            "response": "compose_final_response",
            "confirm": "confirmation_interrupt",
            "execute": "execute_tool",
        },
    )
    graph_builder.add_conditional_edges(
        "confirmation_interrupt",
        route_after_confirmation,
        {"response": "compose_final_response", "execute": "execute_tool"},
    )
    graph_builder.add_conditional_edges(
        "execute_tool",
        route_after_execution,
        {
            "clarify": "clarification_interrupt",
            "decide": "agent_decision",
            "response": "compose_final_response",
        },
    )
    graph_builder.add_conditional_edges(
        "clarification_interrupt",
        route_after_clarification,
        {"execute": "execute_tool", "response": "compose_final_response"},
    )
    graph_builder.add_edge("compose_final_response", "update_long_term_memory")
    graph_builder.add_edge("update_long_term_memory", "finalize_turn")
    graph_builder.add_edge("finalize_turn", END)
    return graph_builder.compile(checkpointer=checkpointer, store=store)


def _build_studio_graph():
    client = build_llm_client()
    return build_graph(llm_client=client)


graph = _build_studio_graph()
