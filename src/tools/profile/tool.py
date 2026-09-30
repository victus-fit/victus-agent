from __future__ import annotations

from tools.contracts import ToolContext, ToolExecution, ToolResult, ToolServices
from tools.profile.contract import ProfileUpdateInput
from tools.profile.policy import decide_with_policy
from tools.profile.validators import validate_profile_update_decision


async def execute(
    input_data: ProfileUpdateInput, context: ToolContext, services: ToolServices
) -> ToolExecution:
    decision = validate_profile_update_decision(decide_with_policy(input_data), input_data)
    if decision.requires_safety_validation:
        status = "blocked"
    elif decision.profile_action == "needs_clarification":
        status = "needs_clarification"
    elif decision.profile_action == "reroute":
        status = "rejected"
    else:
        status = "success"
    if not context.identity.authenticated or not context.identity.subject:
        return ToolExecution(result=ToolResult(status="blocked"))
    if status == "success":
        gateway = services.require("profile_write_gateway")
        category = "restriction" if decision.profile_entity_type == "restriction" else (decision.category or "nutrition")
        try:
            saved = await gateway.update(subject=context.identity.subject, payload={"action": "remove" if decision.profile_action.startswith("remove") else "upsert", "category": category, "label": decision.target or decision.category, "value": decision.target or decision.category, "importance": 5 if decision.restriction_kind == "allergy" else 3, "metadata_json": {"restriction_kind": decision.restriction_kind, "severity": decision.severity, "direction": decision.direction}})
        except Exception as exc:
            return ToolExecution(result=ToolResult(status="error", error={"code":"profile_unavailable","message":str(exc)}))
    return ToolExecution(
        result=ToolResult(
            status=status,
            data={**decision.model_dump(mode="json"), **({"saved": saved} if status == "success" else {})},        ),
    )
