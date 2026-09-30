from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel

from tools.contracts import ToolExecution, ToolExposure, ToolResult
from tools.event_capture.contract import EventCaptureInput
from tools.event_capture.tool import execute as execute_event_capture
from tools.evidence_retrieval.contract import EvidenceRetrievalInput
from tools.evidence_retrieval.tool import execute as execute_evidence_retrieval
from tools.profile.read_contract import ProfileReadInput
from tools.profile.read_tool import execute as execute_profile_read
from tools.profile.contract import ProfileUpdateInput
from tools.profile.tool import execute as execute_profile_update
from tools.diet_plan import DietPlanInput, execute as execute_diet_plan

ToolImplementation = Callable[..., ToolExecution | Any]
ALL_EXPOSURES = frozenset({"langgraph", "mcp", "cli", "test"})


def _description(name: str) -> str:
    return {
        "event_capture": (
            "Use when the user reports a meal or beverage that they consumed. "
            "Provide every consumed item with its numeric quantity and unit; time defaults to "
            "today. Do not use for durable "
            "preferences or restrictions, "
            "future goals or plans, feedback, profile reads, symptoms, biometrics, or lifestyle "
            "metrics."
        ),
        "evidence_retrieval": (
            "Use when the user asks for scientific evidence, studies, or support for a health or "
            "nutrition claim. Search with a focused query and cite the retrieved evidence in the "
            "final answer. Do not use for meal logging, preferences, direct medical diagnosis, or "
            "when a general conversational response is sufficient."
        ),
        "profile": (
            "Use when the user asks about their current diet, latest logged meals, or basic "
            "biometrics such as weight, sleep, energy, or adherence. Choose the narrowest section "
            "that answers the question. Do not use for changing profile data, logging a meal, "
            "or questions about another person."
        ),
        "profile_update": "Use when the user explicitly wants to add, modify, or remove a durable allergy, restriction, or preference.",
        "diet_plan": "Use to create, refine, or activate a personalized diet plan after considering the user's profile and restrictions.",
    }[name]


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    version: str
    description: str
    category: str
    risk: str
    side_effects: bool
    requires_identity: bool
    exposures: frozenset[ToolExposure]
    input_model: type[BaseModel]
    output_model: type[BaseModel]
    implementation: ToolImplementation

    @property
    def input_schema(self) -> dict[str, Any]:
        schema = _inline_local_defs(self.input_model.model_json_schema())
        if self.name == "event_capture":
            _require_event_capture_item_quantity(schema)
        return schema


_DEFINITIONS = (
    (
        "event_capture",
        EventCaptureInput,
        execute_event_capture,
        "capture",
        "high",
        True,
        ALL_EXPOSURES,
    ),
    (
        "evidence_retrieval",
        EvidenceRetrievalInput,
        execute_evidence_retrieval,
        "retrieval",
        "low",
        False,
        frozenset({"langgraph", "test"}),
    ),
    (
        "profile",
        ProfileReadInput,
        execute_profile_read,
        "retrieval",
        "low",
        False,
        frozenset({"langgraph", "test"}),
    ),
    ("profile_update", ProfileUpdateInput, execute_profile_update, "profile", "high", True, frozenset({"langgraph", "test"})),
    ("diet_plan", DietPlanInput, execute_diet_plan, "planning", "high", True, frozenset({"langgraph", "test"})),
)

TOOL_DEFINITIONS = {
    name: ToolDefinition(
        name=name,
        version="1",
        description=_description(name),
        category=category,
        risk=risk,
        side_effects=side_effects,
        requires_identity=True,
        exposures=exposures,
        input_model=input_model,
        output_model=ToolResult,
        implementation=implementation,
    )
    for name, input_model, implementation, category, risk, side_effects, exposures in _DEFINITIONS
}


def list_tools(*, exposure: ToolExposure | None = None) -> list[ToolDefinition]:
    definitions = list(TOOL_DEFINITIONS.values())
    return [item for item in definitions if exposure is None or exposure in item.exposures]


def get_tool(name: str) -> ToolDefinition:
    try:
        return TOOL_DEFINITIONS[name]
    except KeyError as exc:
        raise ValueError(f"unknown tool: {name}") from exc


def _inline_local_defs(schema: dict[str, Any]) -> dict[str, Any]:
    definitions = schema.get("$defs")
    if not isinstance(definitions, dict):
        return schema

    def resolve(value: Any) -> Any:
        if isinstance(value, dict):
            ref = value.get("$ref")
            if isinstance(ref, str) and ref.startswith("#/$defs/"):
                key = ref.rsplit("/", 1)[-1]
                definition = definitions.get(key)
                if isinstance(definition, dict):
                    return resolve(deepcopy(definition))
            return {key: resolve(item) for key, item in value.items() if key != "$defs"}
        if isinstance(value, list):
            return [resolve(item) for item in value]
        return value

    return resolve(schema)


def _require_event_capture_item_quantity(schema: dict[str, Any]) -> None:
    item_schema = (
        schema.get("properties", {})
        .get("items", {})
        .get("items", {})
    )
    if isinstance(item_schema, dict):
        item_schema["required"] = ["name", "quantity", "unit"]
