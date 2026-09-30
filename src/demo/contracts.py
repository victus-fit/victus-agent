from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

DEMO_PROFILE_VERSION = "david-v1"


class DemoContract(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DemoChatRequest(DemoContract):
    conversation_id: str = Field(min_length=1, max_length=200)
    request_id: str = Field(min_length=1, max_length=200)
    message: str = Field(min_length=1, max_length=20_000)
    language: str = Field(default="en", min_length=2, max_length=16)
    demo_state: dict[str, object] | None = None


class DemoChatResponse(DemoContract):
    message: str
    profile_version: Literal[DEMO_PROFILE_VERSION]
    read_only: Literal[True] = True
