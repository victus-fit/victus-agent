from __future__ import annotations

import asyncio

from tools.contracts import ToolContext, ToolIdentity, ToolInvocation, ToolServices
from tools.profile.read_contract import ProfileReadInput
from tools.profile.read_tool import execute
from tools.runtime import ToolRuntime


class RecordingProfileGateway:
    def __init__(self) -> None:
        self.calls: list[dict[str, str]] = []

    async def fetch(self, *, subject: str, section: str) -> dict[str, object]:
        self.calls.append({"subject": subject, "section": section})
        return {"display_name": "David", "biometrics": [{"metric_type": "weight"}]}


def test_profile_tool_reads_only_the_authenticated_subject() -> None:
    gateway = RecordingProfileGateway()
    runtime = ToolRuntime(services=ToolServices({"profile_read_gateway": gateway}))
    result = asyncio.run(
        runtime.invoke_async(
            ToolInvocation(
                name="profile",
                arguments={"section": "biometrics"},
                context=ToolContext(
                    source="test",
                    identity=ToolIdentity(subject="demo:david", authenticated=True),
                ),
            )
        )
    )

    assert result.status == "success"
    assert result.data == {"display_name": "David", "biometrics": [{"metric_type": "weight"}]}
    assert gateway.calls == [{"subject": "demo:david", "section": "biometrics"}]


def test_profile_tool_blocks_anonymous_reads() -> None:
    result = asyncio.run(
        execute(
            input_data=ProfileReadInput(),
            context=ToolContext(source="test"),
            services=ToolServices(),
        )
    )

    assert result.result.status == "blocked"
    assert result.result.error and result.result.error.code == "missing_identity"
