from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx

from tools.event_capture.contract import EventCaptureInput


class DemoMealCaptureUnavailable(RuntimeError):
    pass


@dataclass(frozen=True)
class FullstackDemoMealGateway:
    base_url: str
    api_token: str
    timeout_seconds: float = 10.0
    transport: httpx.AsyncBaseTransport | None = None

    async def capture(self, *, session_id: str, input_data: EventCaptureInput) -> dict[str, Any]:
        if not self.base_url or not self.api_token:
            raise DemoMealCaptureUnavailable("demo meal capture is not configured")
        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds, transport=self.transport) as client:
                response = await client.post(
                    f"{self.base_url.rstrip('/')}/internal/demo/meal-captures",
                    headers={"Authorization": f"Bearer {self.api_token}"},
                    json={"session_id": session_id, **input_data.model_dump(mode="json")},
                )
            response.raise_for_status()
            payload = response.json()
        except httpx.HTTPError as exc:
            raise DemoMealCaptureUnavailable("demo meal capture is unavailable") from exc
        if not isinstance(payload, dict):
            raise DemoMealCaptureUnavailable("demo meal capture returned an invalid response")
        return payload
