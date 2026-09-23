from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class EvidenceFilters(BaseModel):
    model_config = ConfigDict(extra="forbid")

    organism: list[str] | None = Field(default=None, max_length=25)
    evidence_type: list[str] | None = Field(default=None, max_length=25)
    assertion_type: list[str] | None = Field(default=None, max_length=25)

    @field_validator("organism", "evidence_type", "assertion_type")
    @classmethod
    def validate_values(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return value
        if not value or any(not item.strip() for item in value):
            raise ValueError("filters must contain non-empty strings")
        return [item.strip() for item in value]


class EvidenceRetrievalInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(
        min_length=1,
        max_length=4096,
        description="Focused scientific-evidence search query derived from the user's question.",
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=5,
        description="Maximum number of evidence results to return. Keep this at five or fewer.",
    )
    filters: EvidenceFilters | None = Field(
        default=None,
        description="Optional allowed evidence filters. Omit filters unless the user requires them.",
    )

    @field_validator("query")
    @classmethod
    def strip_query(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("query must not be blank")
        return cleaned

    def as_request_payload(self) -> dict[str, Any]:
        return self.model_dump(exclude_none=True)


class RetrievedEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    canonical_evidence_id: str
    paper_id: str | None = None
    evidence_text: str
    source_block_ids: list[str] = Field(default_factory=list)
    evidence_type: str | None = None
    assertion_type: str | None = None
    organism: str | None = None
    population: str | None = None
    intervention_or_exposure: str | None = None
    comparator: str | None = None
    outcomes: list[str] | None = None
    duration: str | None = None


class EvidenceSearchResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rank: int
    score: float
    evidence: RetrievedEvidence


class EvidenceSearchMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    index_version: str
    retrieval_pipeline: str
    took_ms: int


class EvidenceSearchResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: str
    results: list[EvidenceSearchResult]
    metadata: EvidenceSearchMetadata
