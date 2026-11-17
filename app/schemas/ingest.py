from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt, StringConstraints, field_validator

DocId = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")]
Group = Annotated[str, StringConstraints(min_length=1, max_length=32)]


class DocumentManifest(BaseModel):
    """One versioned document. There is deliberately NO tenant_id here: the tenant comes from the API key."""

    model_config = ConfigDict(extra="forbid")

    doc_id: DocId
    version: Annotated[StrictInt, Field(ge=1, le=1_000_000)]
    kind: Literal["kb", "ticket", "log"]
    title: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=200)]
    content: Annotated[str, StringConstraints(min_length=1, max_length=100_000)]
    topic: Annotated[str, StringConstraints(max_length=64)] | None = None
    visibility: Literal["tenant", "restricted"] = "tenant"
    acl_groups: list[Group] = Field(default_factory=list, max_length=16)
    effective_date: date | None = None
    source_uri: Annotated[str, StringConstraints(max_length=300)] | None = None

    @field_validator("content", "title")
    @classmethod
    def _no_nul(cls, v: str) -> str:
        if "\x00" in v:
            raise ValueError("NUL characters are not allowed")
        return v

    @field_validator("source_uri")
    @classmethod
    def _scheme(cls, v: str | None) -> str | None:
        if v is not None and not v.startswith(("landing://", "file://")):
            raise ValueError("source_uri must start with landing:// or file://")
        return v


class IngestResponse(BaseModel):
    outcome: str
    doc_id: str
    version: int
    active_version: int | None
    status: str


@dataclass(frozen=True)
class RunContext:
    """Orchestrator lineage (from X-Run-Id etc. headers). Recorded only - never used for decisions."""

    run_id: str | None = None
    dag_id: str | None = None
    task_id: str | None = None
    try_number: int | None = None
    data_interval_start: str | None = None
