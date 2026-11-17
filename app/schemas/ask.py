from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

Question = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=1000)]

ABSTAIN_TEXT = "I don't have enough evidence in the documents available to you to answer that."


class AskRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: Question
    top_k: int = Field(default=5, ge=1, le=10)


# ---- what the LLM must emit (also the fine-tune target format) -------------------------

class LLMCitation(BaseModel):
    source_id: str = Field(max_length=16)  # alias such as "S1"
    quote: str = Field(max_length=400)


class GroundedAnswer(BaseModel):
    answer: str = Field(default="", max_length=1500)
    abstain: bool = False
    conflict: bool = False
    citations: list[LLMCitation] = Field(default_factory=list, max_length=8)


def answer_json_schema(aliases: list[str]) -> dict:
    """JSON Schema handed to Ollama ``format`` — source_id is restricted to this request's aliases."""
    schema = GroundedAnswer.model_json_schema()
    schema["$defs"]["LLMCitation"]["properties"]["source_id"] = {"type": "string", "enum": aliases}
    return schema


# ---- what the API returns -----------------------------------------------------------------

class Citation(BaseModel):
    source_id: str
    doc_id: str
    version: int
    chunk_id: str
    content_sha256: str
    source_uri: str | None
    excerpt: str


class ConflictInfo(BaseModel):
    detected: bool = False
    doc_ids: list[str] = Field(default_factory=list)


class AskResponse(BaseModel):
    status: Literal["answered", "abstained", "conflict"]
    answer: str
    reason: Literal["insufficient_evidence", "unverifiable_output"] | None = None
    citations: list[Citation] = Field(default_factory=list)
    conflict: ConflictInfo = Field(default_factory=ConflictInfo)
    model: str
    request_id: str
