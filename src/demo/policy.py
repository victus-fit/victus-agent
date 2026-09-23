from __future__ import annotations

from typing import Final


_DENIED_INPUT_MARKERS: Final = (
    "system prompt",
    "ignore previous",
    "ignore the rules",
    "reveal your instructions",
    "api key",
    "credential",
    "password",
    "access token",
    "other user",
    "otra persona",
    "cambiar perfil",
    "change profile",
    "update profile",
    "webhook",
    "database write",
)


def is_disallowed_demo_input(message: str) -> bool:
    """Reject attempts to cross the public-demo trust boundary before graph execution."""
    value = message.casefold()
    return any(marker in value for marker in _DENIED_INPUT_MARKERS)


def denied_demo_message(language: str) -> str:
    if language.casefold().startswith("es"):
        return (
            "La demo usa el perfil fijo de David y una sesión temporal. No puedo cambiar su "
            "identidad, acceder a otros datos ni revelar información interna."
        )
    return (
        "This demo uses David's fixed profile and a temporary session. I cannot change his "
        "identity, access other data, or reveal internal information."
    )
