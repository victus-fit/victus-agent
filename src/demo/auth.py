from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass
from typing import Any

import jwt


class DemoTokenError(Exception):
    status_code = 401


class DemoTokenForbidden(DemoTokenError):
    status_code = 403


class UnsupportedDemoProfile(DemoTokenError):
    status_code = 422


class DemoAuthConfigurationError(Exception):
    pass


@dataclass(frozen=True)
class DemoIdentity:
    subject: str
    profile_version: str
    session_id: str


class InMemoryReplayProtector:
    """Single-process replay protection for one-time, <=30 second demo JWTs."""

    def __init__(self) -> None:
        self._used: dict[str, float] = {}
        self._lock = threading.Lock()

    def consume(self, jti: str, expires_at: float) -> bool:
        now = time.time()
        with self._lock:
            self._used = {key: expiry for key, expiry in self._used.items() if expiry > now}
            if jti in self._used:
                return False
            self._used[jti] = expires_at
            return True


class DemoTokenVerifier:
    def __init__(self, *, jwks: dict[str, Any], replay_protector: InMemoryReplayProtector | None = None):
        keys = jwks.get("keys") if isinstance(jwks, dict) else None
        if not isinstance(keys, list):
            raise DemoAuthConfigurationError("demo JWKS must contain keys")
        self._keys = {
            key.get("kid"): jwt.PyJWK.from_dict(key).key
            for key in keys
            if isinstance(key, dict) and isinstance(key.get("kid"), str)
        }
        if not self._keys:
            raise DemoAuthConfigurationError("demo JWKS must contain keyed public keys")
        self._replay_protector = replay_protector or InMemoryReplayProtector()

    @classmethod
    def from_environment(cls) -> "DemoTokenVerifier | None":
        raw_jwks = os.getenv("VICTUS_DEMO_JWT_JWKS_JSON", "").strip()
        if not raw_jwks:
            return None
        try:
            return cls(jwks=json.loads(raw_jwks))
        except (ValueError, jwt.PyJWTError) as exc:
            raise DemoAuthConfigurationError("demo JWT configuration is invalid") from exc

    def verify(self, token: str, *, profile_version: str) -> DemoIdentity:
        try:
            header = jwt.get_unverified_header(token)
            kid = header.get("kid")
            if not isinstance(kid, str) or kid not in self._keys:
                raise DemoTokenError("unknown demo signing key")
            claims = jwt.decode(
                token,
                self._keys[kid],
                algorithms=["ES256"],
                audience="victus-agent",
                issuer="victus-webapp",
                options={"require": ["aud", "exp", "iat", "iss", "jti", "sub", "sid"]},
            )
        except DemoTokenError:
            raise
        except jwt.PyJWTError as exc:
            raise DemoTokenError("invalid demo token") from exc

        if claims.get("sub") != "demo:david":
            raise DemoTokenError("invalid demo subject")
        if claims.get("demo") is not True:
            raise DemoTokenForbidden("demo claim is required")
        scope = claims.get("scope")
        if not isinstance(scope, list) or not {"demo:chat", "demo:read"}.issubset(scope):
            raise DemoTokenForbidden("demo scope is required")
        if claims.get("profile_version") != profile_version:
            raise UnsupportedDemoProfile("unsupported demo profile")
        session_id = claims.get("sid")
        if not isinstance(session_id, str) or not session_id or len(session_id) > 200:
            raise DemoTokenError("invalid demo session")

        issued_at, expires_at = claims.get("iat"), claims.get("exp")
        if (
            isinstance(issued_at, bool)
            or isinstance(expires_at, bool)
            or not isinstance(issued_at, (int, float))
            or not isinstance(expires_at, (int, float))
            or expires_at - issued_at > 30
        ):
            raise DemoTokenError("demo token lifetime is invalid")
        jti = claims.get("jti")
        if not isinstance(jti, str) or not jti or not self._replay_protector.consume(jti, float(expires_at)):
            raise DemoTokenError("demo token has already been used")
        return DemoIdentity(
            subject="demo:david",
            profile_version=profile_version,
            session_id=session_id,
        )
