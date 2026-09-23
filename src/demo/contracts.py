from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class DemoContract(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DemoMeal(DemoContract):
    name: str = Field(min_length=1, max_length=100)
    calories: int = Field(ge=0, le=5_000)
    protein_g: int = Field(ge=0, le=500)
    carbohydrates_g: int = Field(ge=0, le=1_000)
    fat_g: int = Field(ge=0, le=500)


class DemoDay(DemoContract):
    day: str = Field(min_length=1, max_length=20)
    meals: list[DemoMeal] = Field(min_length=1, max_length=8)
    calories: int = Field(ge=0, le=10_000)
    protein_g: int = Field(ge=0, le=1_000)
    carbohydrates_g: int = Field(ge=0, le=2_000)
    fat_g: int = Field(ge=0, le=1_000)

    @model_validator(mode="after")
    def totals_match_meals(self) -> "DemoDay":
        for field in ("calories", "protein_g", "carbohydrates_g", "fat_g"):
            if sum(getattr(meal, field) for meal in self.meals) != getattr(self, field):
                raise ValueError(f"{field} must equal the sum of meals")
        return self


class DemoFixture(DemoContract):
    profile_version: Literal["david-v1"]
    read_only: Literal[True]
    display_name: str
    objective: str
    preferences: list[str]
    restrictions: list[str]
    routine: str
    biometrics: dict[str, str | int | float]
    nutrition_targets: dict[str, int]
    current_diet: list[DemoDay] = Field(min_length=7, max_length=7)


class DemoChatRequest(DemoContract):
    conversation_id: str = Field(min_length=1, max_length=200)
    request_id: str = Field(min_length=1, max_length=200)
    message: str = Field(min_length=1, max_length=20_000)
    language: str = Field(default="en", min_length=2, max_length=16)


class DemoChatResponse(DemoContract):
    message: str
    profile_version: Literal["david-v1"]
    read_only: Literal[True] = True
