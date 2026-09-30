from __future__ import annotations
from typing import Any, Literal
from pydantic import Field
from tools.contracts import ContractModel, ToolContext, ToolExecution, ToolResult, ToolServices

class DietPlanInput(ContractModel):
    action: Literal["create_draft", "refine", "activate"]
    plan_id: str | None = None
    plan_json: dict[str, Any] | None = Field(default=None, description="Structured diet proposal with goals, meals, foods, portions, and alternatives.")

async def execute(input_data: DietPlanInput, context: ToolContext, services: ToolServices) -> ToolExecution:
    if not context.identity.authenticated or not context.identity.subject: return ToolExecution(result=ToolResult(status="blocked"))
    if input_data.action != "activate" and not input_data.plan_json: return ToolExecution(result=ToolResult(status="needs_clarification", data={"question":"Necesito el borrador de dieta para guardarlo."}))
    gateway = services.require("diet_plan_gateway")
    try:
        saved = await gateway.diet_plan(subject=context.identity.subject, payload=input_data.model_dump(mode="json"))
    except Exception as exc:
        return ToolExecution(result=ToolResult(status="error", error={"code":"diet_plan_unavailable","message":str(exc)}))
    return ToolExecution(result=ToolResult(status="success", data=saved))
