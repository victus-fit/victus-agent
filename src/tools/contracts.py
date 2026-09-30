from __future__ import annotations

from dataclasses import dataclass, field
from collections.abc import Mapping
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from domain.events.base import SafetyStatus
from domain.events.refs import ToolEventRef

ToolStatus = Literal["success", "needs_clarification", "blocked", "rejected", "error"]
ToolExposure = Literal["langgraph", "mcp", "cli", "test"]
ExpectedAnswerType = Literal[
    "quantity",
    "time",
    "meal_reference",
    "preference_strength",
    "restriction_type",
    "goal_target",
    "yes_no",
    "free_text",
]


class ContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ClarificationRequest(ContractModel):
    missing_fields: list[str]
    question: str
    expected_answer_type: ExpectedAnswerType
    resume_node: str | None = None
    resume_action: str | None = None


class ToolSafety(ContractModel):
    status: SafetyStatus
    reasons: list[str] = Field(default_factory=list)


class ToolError(ContractModel):
    code: str
    message: str


class ToolResult(ContractModel):
    status: ToolStatus
    data: Any | None = None
    events_emitted: list[ToolEventRef] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    clarification: ClarificationRequest | None = None
    safety: ToolSafety = Field(default_factory=lambda: ToolSafety(status="ok"))
    error: ToolError | None = None


class ToolIdentity(ContractModel):
    subject: str | None = None
    authenticated: bool = False
    permissions: frozenset[str] = Field(default_factory=frozenset)


class ToolContext(ContractModel):
    source: ToolExposure
    identity: ToolIdentity = Field(default_factory=ToolIdentity)
    original_text: str | None = None
    trace_id: str | None = None
    idempotency_key: str | None = None
    session_id: str | None = None


class ToolInvocation(ContractModel):
    name: str
    arguments: dict[str, Any]
    context: ToolContext


@dataclass(frozen=True)
class ToolServices:
    values: Mapping[str, Any] = field(default_factory=dict)

    def require(self, name: str) -> Any:
        try:
            return self.values[name]
        except KeyError as exc:
            raise RuntimeError(f"missing tool service: {name}") from exc

    def get(self, name: str) -> Any | None:
        return self.values.get(name)


@dataclass(frozen=True)
class ToolExecution:
    result: ToolResult
    events: tuple[Any, ...] = field(default_factory=tuple)
