from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import httpx

@dataclass(frozen=True)
class WebAppProfileWriteGateway:
    base_url: str = ""
    api_token: str = ""
    async def update(self, *, subject: str, payload: dict[str, Any]) -> dict[str, Any]:
        if not self.base_url or not self.api_token: raise RuntimeError("profile update is not configured")
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.post(f"{self.base_url.rstrip('/')}/internal/agent/profile/update", headers={"Authorization": f"Bearer {self.api_token}"}, json={"subject": subject, **payload})
        response.raise_for_status(); result = response.json()
        if not isinstance(result, dict): raise RuntimeError("profile update returned an invalid response")
        return result

    async def diet_plan(self, *, subject: str, payload: dict[str, Any]) -> dict[str, Any]:
        if not self.base_url or not self.api_token: raise RuntimeError("diet plan is not configured")
        async with httpx.AsyncClient(timeout=20) as client:
            response = await client.post(f"{self.base_url.rstrip('/')}/internal/agent/diet-plans", headers={"Authorization": f"Bearer {self.api_token}"}, json={"subject": subject, **payload})
        response.raise_for_status(); result = response.json()
        if not isinstance(result, dict): raise RuntimeError("diet plan returned an invalid response")
        return result
