from __future__ import annotations

from typing import Annotated, Any, Literal, TypedDict

from langgraph.graph.message import add_messages

from domain.events.refs import ToolEventRef

GRAPH_VERSION = "1"


class RequestState(TypedDict, total=False):
    request_id: str
    user_id: str
    original_text: str
    working_text: str
    received_at: str
    locale: str
    timezone: str
    conversation_id: str
    execution_mode: str
    demo_profile: dict[str, Any]


class SafetyState(TypedDict, total=False):
    status: Literal["unknown", "ok", "warning", "blocked", "needs_clarification"]
    reasons: list[str]
    decision: str
    severity: str
    categories: list[str]


class ToolContextState(TypedDict, total=False):
    allowed_tools: list[str]
    proposed_action: dict[str, Any]
    last_tool_result: dict[str, Any]
    pending_clarification: dict[str, Any]
    tool_results: list[dict[str, Any]]
    loop_count: int
    confirmation: dict[str, Any]


class PlanningState(TypedDict, total=False):
    session_id: str
    revision_id: str
    artifact_id: str
    candidate_artifact: dict[str, Any]
    validation_report: dict[str, Any]


class EvidenceState(TypedDict, total=False):
    query: str
    retrieved_evidence: list[Any]
    cited_evidence: list[Any]
    generated_claims: list[Any]


class ClarificationState(TypedDict, total=False):
    clarification_id: str
    missing_fields: list[str]
    question: str
    expected_answer_type: str
    resume_node: str
    resume_action: str


class ResponseState(TypedDict, total=False):
    mode: Literal["final", "clarification", "blocked", "error"]
    user_message: str


class MemoryState(TypedDict, total=False):
    recalled: list[dict[str, Any]]
    compact_summary: str


class AuditState(TypedDict):
    node_path: list[str]
    events_emitted: list[ToolEventRef]
    warnings: list[str]
    errors: list[str]
    transforms: list[dict[str, str]]


class VictusGraphState(TypedDict, total=False):
    messages: Annotated[list[Any], add_messages]
    request: RequestState
    safety: SafetyState
    tool_context: ToolContextState
    planning: PlanningState
    evidence: EvidenceState
    clarification: ClarificationState
    response: ResponseState
    memory: MemoryState
    graph_version: str
    audit: AuditState
