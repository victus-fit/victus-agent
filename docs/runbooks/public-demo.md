# Public demo runbook

## Configure the public key

The WebApp owns the ES256 private key and signs a JWT with a `kid`. Export only its public JWKS into
the agent process:

```bash
VICTUS_DEMO_JWT_JWKS_JSON='{"keys":[{"kty":"EC","crv":"P-256","x":"...","y":"...","kid":"demo-es256-2026-01","alg":"ES256","use":"sig"}]}'
```

Do not set a private key, an HS256 secret, or a remote JWKS URL in the agent. To rotate a key, publish
both public keys, sign new tokens with the new `kid`, then remove the retired key only after all
30-second tokens have expired.

## Configure ephemeral sessions and meal state

The demo uses the same model configuration as `/chat`. Its checkpoints and memory are process-local
and expire automatically, while temporary meals and biometrics are owned by the WebApp's demo store.
Configure a bounded graph-session TTL and the authenticated internal WebApp endpoint:

```bash
VICTUS_DEMO_SESSION_TTL_SECONDS=900
VICTUS_DEMO_STATE_API_URL=http://victus-backend:8000
VICTUS_DEMO_STATE_API_TOKEN=shared-demo-agent-token
```

The allowed range is 1–3600 seconds. The default is 900 seconds. Set the same value as
`VICTUS_DEMO_AGENT_API_TOKEN` in the WebApp environment; the endpoint is for agent-to-WebApp
traffic only. Do not deploy
multiple agent replicas with this in-memory checkpoint implementation; move the session and
JWT-replay stores to shared TTL storage first.

## WebApp token requirement

For each browser page session, the WebApp must create an opaque random `sid`, retain it only for that
browser session, and include it as the JWT `sid` claim on every request. It must mint a new short-lived
JWT and `jti` per request. A page reload creates a new `sid`; the agent never uses the browser's
`conversation_id` to look up demo state.

## Validate

```bash
uv run --extra test pytest tests/test_public_demo.py -q
```

The test suite verifies ES256 audience, expiry, scope, profile-version, signed session, and replay
denials. It also verifies that the standard graph exposes the demo-approved tools, including
read-only `profile`, and that `event_capture` completes through the temporary WebApp store. A valid demo token must be newly minted
for every request because its `jti` is single use.

## Persisted David profile

The WebApp owns the seeded `demo:david` profile. Update it through the WebApp's initialization and
data migrations; do not reintroduce a static profile fixture in the agent. The agent's `profile`
tool can read only David's latest logged diet and basic biometrics through
`/internal/agent/profile`, protected by `VICTUS_DEMO_STATE_API_TOKEN`.
