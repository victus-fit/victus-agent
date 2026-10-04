from __future__ import annotations
from typing import Literal
from pydantic import Field, model_validator
from tools.contracts import ContractModel, ToolContext, ToolExecution, ToolResult, ToolServices

WEEK_DAYS = frozenset({"Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"})


class DietPlanTargets(ContractModel):
    calories_kcal: float = Field(gt=0)
    protein_g: float = Field(gt=0)
    carbohydrate_g: float = Field(gt=0)
    fat_g: float = Field(gt=0)


class DietPlanFood(ContractModel):
    name: str = Field(min_length=1)
    portion: str = Field(min_length=1, description="Serving amount, for example '180 g' or '2 unidades'.")


class DietPlanMeal(ContractModel):
    name: str = Field(min_length=1, description="Named meal, for example 'Desayuno' or 'Cena'.")
    food_items: list[DietPlanFood] = Field(min_length=1)


class DietPlanDay(ContractModel):
    day: str = Field(description="One unique Spanish weekday from Lunes through Domingo.")
    focus: str = Field(min_length=1)
    calories: float = Field(gt=0)
    meals: list[DietPlanMeal] = Field(min_length=2, max_length=3)


class DietPlanDocument(ContractModel):
    description: str = Field(min_length=1, description="Brief explanation of the complete weekly proposal.")
    targets: DietPlanTargets = Field(description="Positive daily calorie and macronutrient targets.")
    days: list[DietPlanDay] = Field(
        min_length=7,
        max_length=7,
        description="Exactly one complete daily plan for each Spanish weekday, Lunes through Domingo.",
    )

    @model_validator(mode="after")
    def validate_week(self) -> "DietPlanDocument":
        days = [day.day for day in self.days]
        if set(days) != WEEK_DAYS or len(set(days)) != 7:
            raise ValueError("days must contain each Spanish weekday exactly once")
        return self


class DietPlanInput(ContractModel):
    action: Literal["create", "refine", "activate"] = Field(
        description="create saves a new draft, refine revises a draft, activate selects an accepted draft as active."
    )
    plan_id: str | None = Field(default=None, description="Required for refine or activate; omit for create.")
    plan_json: DietPlanDocument | None = Field(
        default=None,
        description="Required for create or refine. A full seven-day plan with daily targets and two or three meals per day; never a partial plan.",
    )

    @model_validator(mode="after")
    def validate_action_payload(self) -> "DietPlanInput":
        if self.action in {"refine", "activate"} and not self.plan_id:
            raise ValueError("plan_id is required for refine or activate")
        if self.action in {"create", "refine"} and not self.plan_json:
            raise ValueError("plan_json is required for create or refine")
        if self.action == "activate" and self.plan_json is not None:
            raise ValueError("plan_json is not accepted for activate")
        return self

async def execute(input_data: DietPlanInput, context: ToolContext, services: ToolServices) -> ToolExecution:
    if not context.identity.authenticated or not context.identity.subject: return ToolExecution(result=ToolResult(status="blocked"))
    if input_data.action != "activate" and not input_data.plan_json: return ToolExecution(result=ToolResult(status="needs_clarification", data={"question":"Necesito el borrador de dieta para guardarlo."}))
    gateway = services.require("diet_plan_gateway")
    try:
        saved = await gateway.diet_plan(subject=context.identity.subject, payload=input_data.model_dump(mode="json"))
    except Exception as exc:
        return ToolExecution(result=ToolResult(status="error", error={"code":"diet_plan_unavailable","message":str(exc)}))
    return ToolExecution(result=ToolResult(status="success", data=saved))
