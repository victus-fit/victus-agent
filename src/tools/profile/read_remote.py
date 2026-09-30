from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx

from tools.profile.read_contract import ProfileSection


class ProfileReadUnavailable(RuntimeError):
    pass


@dataclass(frozen=True)
class WebAppProfileGateway:
    base_url: str = ""
    api_token: str = ""
    timeout_seconds: float = 10.0
    transport: httpx.AsyncBaseTransport | None = None

    async def fetch(self, *, subject: str, section: ProfileSection) -> dict[str, Any]:
        if not self.base_url or not self.api_token:
            raise ProfileReadUnavailable("profile lookup is not configured")
        try:
            async with httpx.AsyncClient(
                timeout=self.timeout_seconds, transport=self.transport
            ) as client:
                response = await client.post(
                    f"{self.base_url.rstrip('/')}/internal/agent/profile",
                    headers={"Authorization": f"Bearer {self.api_token}"},
                    json={"subject": subject, "section": section},
                )
        except httpx.HTTPError as exc:
            raise ProfileReadUnavailable("profile lookup is unavailable") from exc
        if response.status_code == 404:
            raise ProfileReadUnavailable("profile is unavailable")
        if response.status_code in {401, 422} or response.status_code >= 500:
            raise ProfileReadUnavailable("profile lookup is unavailable")
        try:
            response.raise_for_status()
            payload = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise ProfileReadUnavailable("profile lookup returned an invalid response") from exc
        if not isinstance(payload, dict):
            raise ProfileReadUnavailable("profile lookup returned an invalid response")
        return payload
