from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx

from tools.event_capture.contract import EventCaptureInput


@dataclass(frozen=True)
class WebAppMealCaptureGateway:
    base_url: str = ""
    api_token: str = ""

    async def capture(self, *, subject: str, input_data: EventCaptureInput) -> dict[str, Any]:
        if not self.base_url or not self.api_token:
            raise RuntimeError("meal capture is not configured")
        async with httpx.AsyncClient(timeout=15) as client:
            response = await client.post(
                f"{self.base_url.rstrip('/')}/internal/agent/meal-captures",
                headers={"Authorization": f"Bearer {self.api_token}"},
                json={"subject": subject, **input_data.model_dump(mode="json")},
            )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise RuntimeError("meal capture returned an invalid response")
        return payload
