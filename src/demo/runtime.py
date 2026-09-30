from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass, field
from typing import Any

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore

from adapters.langgraph.engine.graph import build_graph
from adapters.langgraph.runtime.context import AGENT_ENABLED_TOOLS
from bootstrap.runtime import safety_precheck
from tools.contracts import ToolServices
from tools.event_capture.demo_remote import FullstackDemoMealGateway
from tools.evidence_retrieval.remote import VictusRAGEvidenceGateway
from tools.profile.read_remote import WebAppProfileGateway
from tools.runtime import ToolRuntime
from victus_platform.llm.contracts import LLMClient
from victus_platform.telemetry import new_trace_id

DEMO_ENABLED_TOOLS = AGENT_ENABLED_TOOLS
DEFAULT_SESSION_TTL_SECONDS = 15 * 60
MAX_SESSION_TTL_SECONDS = 60 * 60


class DemoRuntimeConfigurationError(RuntimeError):
    pass


@dataclass
class _DemoSession:
    graph: Any
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    last_accessed_at: float = field(default_factory=time.monotonic)


class DemoSessionManager:
    """Build isolated graph resources per signed public-demo session with a bounded lifetime."""

    def __init__(
        self,
        *,
        llm_client: LLMClient,
        session_ttl_seconds: int | None = None,
        demo_meal_capture_gateway: FullstackDemoMealGateway | None = None,
        profile_read_gateway: WebAppProfileGateway | None = None,
    ) -> None:
        self._llm_client = llm_client
        self._ttl_seconds = session_ttl_seconds or _session_ttl_from_environment()
        self._demo_meal_capture_gateway = demo_meal_capture_gateway or FullstackDemoMealGateway(
            base_url=os.getenv("VICTUS_DEMO_STATE_API_URL", ""),
            api_token=os.getenv("VICTUS_DEMO_STATE_API_TOKEN", ""),
        )
        self._profile_read_gateway = profile_read_gateway or WebAppProfileGateway(
            base_url=os.getenv("VICTUS_DEMO_STATE_API_URL", ""),
            api_token=os.getenv("VICTUS_DEMO_STATE_API_TOKEN", ""),
        )
        self._sessions: dict[str, _DemoSession] = {}
        self._lock = asyncio.Lock()

    async def invoke(self, *, session_id: str, graph_input: dict[str, Any]) -> dict[str, Any]:
        session = await self._session(session_id)
        async with session.lock:
            session.last_accessed_at = time.monotonic()
            config = {
                "configurable": {
                    "thread_id": session_id,
                    "user_id": str(graph_input["request"]["user_id"]),
                }
            }
            return await session.graph.ainvoke(graph_input, config=config)

    async def _session(self, session_id: str) -> _DemoSession:
        async with self._lock:
            now = time.monotonic()
            self._sessions = {
                key: value
                for key, value in self._sessions.items()
                if now - value.last_accessed_at < self._ttl_seconds
            }
            existing = self._sessions.get(session_id)
            if existing is not None:
                return existing
            runtime = _build_demo_runtime(
                self._demo_meal_capture_gateway, self._profile_read_gateway
            )
            session = _DemoSession(
                graph=build_graph(
                    llm_client=self._llm_client,
                    tool_runtime=runtime,
                    checkpointer=InMemorySaver(),
                    store=InMemoryStore(),
                    enabled_tools=DEMO_ENABLED_TOOLS,
                )
            )
            self._sessions[session_id] = session
            return session


def _build_demo_runtime(
    demo_meal_capture_gateway: FullstackDemoMealGateway,
    profile_read_gateway: WebAppProfileGateway,
) -> ToolRuntime:
    return ToolRuntime(
        services=ToolServices(
            {
                "demo_meal_capture_gateway": demo_meal_capture_gateway,
                "profile_read_gateway": profile_read_gateway,
                "evidence_retrieval_gateway": VictusRAGEvidenceGateway(
                    base_url=os.getenv("VICTUS_RAG_API_URL", ""),
                    api_token=os.getenv("VICTUS_RAG_API_TOKEN", ""),
                    timeout_seconds=float(os.getenv("VICTUS_RAG_API_TIMEOUT_SECONDS", "10")),
                )
            }
        ),
        trace_id_factory=new_trace_id,
        precheck=safety_precheck,
    )


def _session_ttl_from_environment() -> int:
    raw = os.getenv("VICTUS_DEMO_SESSION_TTL_SECONDS", str(DEFAULT_SESSION_TTL_SECONDS))
    try:
        value = int(raw)
    except ValueError as exc:
        raise DemoRuntimeConfigurationError("VICTUS_DEMO_SESSION_TTL_SECONDS must be an integer") from exc
    if not 1 <= value <= MAX_SESSION_TTL_SECONDS:
        raise DemoRuntimeConfigurationError("demo session TTL must be between 1 and 3600 seconds")
    return value
