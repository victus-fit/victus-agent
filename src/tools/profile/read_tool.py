from __future__ import annotations

from typing import Protocol, cast

from tools.contracts import ToolError, ToolExecution, ToolResult, ToolServices
from tools.profile.read_contract import ProfileReadInput, ProfileSection
from tools.profile.read_remote import ProfileReadUnavailable


class ProfileReadGateway(Protocol):
    async def fetch(self, *, subject: str, section: ProfileSection) -> dict[str, object]: ...


async def execute(
    input_data: ProfileReadInput, context, services: ToolServices
) -> ToolExecution:
    subject = context.identity.subject
    if not context.identity.authenticated or not subject:
        return ToolExecution(
            result=ToolResult(
                status="blocked",
                error=ToolError(code="missing_identity", message="profile requires an authenticated user"),
            )
        )
    gateway = cast(ProfileReadGateway, services.require("profile_read_gateway"))
    try:
        profile = await gateway.fetch(subject=subject, section=input_data.section)
    except ProfileReadUnavailable as exc:
        return ToolExecution(
            result=ToolResult(
                status="error", error=ToolError(code="profile_unavailable", message=str(exc))
            )
        )
    return ToolExecution(result=ToolResult(status="success", data=profile))
