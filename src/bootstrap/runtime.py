from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Iterator

from tools.runtime import ToolRuntime
from tools.contracts import ToolResult, ToolSafety, ToolServices
from tools.evidence_retrieval.remote import VictusRAGEvidenceGateway
from victus_platform.identity.profile_gateway import BackendProfileGateway
from victus_platform.safety.rules import SafetyPrecheck, SafetyPrecheckInput
from victus_platform.telemetry import new_trace_id


@contextmanager
def event_store_scope() -> Iterator[object | None]:
    if not os.getenv("DATABASE_URL"):
        yield None
        return

    from victus_platform.database.engine import build_engine
    from victus_platform.repositories.events import PostgresEventStore
    from victus_platform.repositories.projections import ProjectionRepository

    engine = build_engine()
    with engine.begin() as connection:
        yield _ProjectingEventStore(
            PostgresEventStore(connection),
            ProjectionRepository(connection),
        )


@contextmanager
def projection_repository_scope() -> Iterator[object | None]:
    if not os.getenv("DATABASE_URL"):
        yield None
        return

    from victus_platform.database.engine import build_engine
    from victus_platform.repositories.projections import ProjectionRepository

    engine = build_engine()
    with engine.begin() as connection:
        yield ProjectionRepository(connection)


def build_runtime() -> ToolRuntime:
    return ToolRuntime(
        event_store_scope=event_store_scope,
        services=ToolServices(
            {
                "profile_gateway": BackendProfileGateway(),
                "evidence_retrieval_gateway": VictusRAGEvidenceGateway(
                    base_url=os.getenv("VICTUS_RAG_API_URL", ""),
                    api_token=os.getenv("VICTUS_RAG_API_TOKEN", ""),
                    timeout_seconds=float(os.getenv("VICTUS_RAG_API_TIMEOUT_SECONDS", "10")),
                ),
            }
        ),
        trace_id_factory=new_trace_id,
        precheck=safety_precheck,
    )


def safety_precheck(input_data, context) -> ToolResult | None:
    text = getattr(input_data, "normalized_text", None)
    if not isinstance(text, str) or not text:
        return None
    result = SafetyPrecheck().check(
        SafetyPrecheckInput(original_text=text, working_text=text)
    )
    if result.decision == "allow":
        return None
    return ToolResult(
        status="blocked",
        safety=ToolSafety(status="blocked", reasons=result.reasons),
    )


class _ProjectingEventStore:
    def __init__(self, event_store, projections) -> None:
        self._event_store = event_store
        self._projections = projections

    def append(self, event):
        from domain.projections.registry import PROJECTION_REGISTRY

        appended = self._event_store.append(event)
        loaders = {
            "user_profile": self._projections.get_user_profile,
            "constraint": self._projections.get_constraint,
            "nutrition_status": self._projections.get_nutrition_status,
            "planning_history": self._projections.get_planning_history,
        }
        savers = {
            "user_profile": self._projections.save_user_profile,
            "constraint": self._projections.save_constraint,
            "nutrition_status": self._projections.save_nutrition_status,
            "planning_history": self._projections.save_planning_history,
        }
        for name, definition in PROJECTION_REGISTRY.items():
            if appended.event_type not in definition.events:
                continue
            projection = definition.apply(loaders[name](appended.user_id), appended)
            savers[name](projection)
            sequence = getattr(projection, "last_event_seq", None)
            if sequence is None:
                sequence = projection.derived_from_event_seq
            self._projections.save_offset(name, sequence)
        return appended
