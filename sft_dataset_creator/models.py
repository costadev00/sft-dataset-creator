from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Section(StrictModel):
    id: str
    text: str
    title: str | None = None


class Document(StrictModel):
    id: str
    text: str
    title: str | None = None
    sections: list[Section] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    license: str | None = None
    source: str

    def evidence_sections(self) -> list[Section]:
        return self.sections or [Section(id="0", title=self.title, text=self.text)]


class DocumentIndex(StrictModel):
    id: str
    source: str
    title: str | None = None
    estimated_tokens: int
    stratum: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvidenceSpan(StrictModel):
    document_id: str
    section_id: str = "0"
    start: int = Field(ge=0)
    end: int = Field(ge=0)
    quote: str | None = None

    @model_validator(mode="after")
    def validate_span(self) -> "EvidenceSpan":
        if self.end <= self.start:
            raise ValueError("evidence end must be greater than start")
        return self


class ChatMessage(StrictModel):
    role: Literal["system", "user", "assistant"]
    content: str


class PlannedSlot(StrictModel):
    id: str
    document_id: str
    task: str
    difficulty: str
    ordinal: int
    chunk_id: str | None = None


class DatasetPlan(StrictModel):
    version: str = "2"
    project_name: str
    run_id: str
    config_hash: str
    seed: int
    created_at: str = Field(default_factory=utc_now)
    corpus_snapshot: str
    database_path: str | None = None
    documents: list[DocumentIndex] = Field(default_factory=list)
    reserve_document_ids: list[str] = Field(default_factory=list)
    slots: list[PlannedSlot] = Field(default_factory=list)
    estimates: dict[str, Any] = Field(default_factory=dict)


class GenerationRequest(StrictModel):
    request_id: str | None = None
    seed: int | None = None
    slot_id: str
    document_id: str
    task: str
    difficulty: str
    messages: list[ChatMessage]
    response_schema: dict[str, Any]
    max_output_tokens: int
    metadata: dict[str, Any] = Field(default_factory=dict)


class BackendResponse(StrictModel):
    payload: dict[str, Any]
    raw_text: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    backend: str
    model: str


class BatchGenerationResult(StrictModel):
    request_id: str
    response: BackendResponse | None = None
    error: str | None = None
    latency_seconds: float = Field(default=0.0, ge=0.0)
    queue_depth: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def exactly_one_outcome(self) -> "BatchGenerationResult":
        if (self.response is None) == (self.error is None):
            raise ValueError("batch result must contain exactly one of response or error")
        return self


class SFTCandidate(StrictModel):
    id: str
    slot_id: str
    attempt: int = Field(ge=1)
    source: str
    document_id: str
    source_title: str | None = None
    task: str
    difficulty: str
    instruction: str
    input: str = ""
    output: str
    messages: list[ChatMessage]
    evidence: list[EvidenceSpan]
    metadata: dict[str, Any] = Field(default_factory=dict)
    generator: str
    model: str


class EvaluationResult(StrictModel):
    candidate_id: str
    verdict: Literal["accept", "reject", "review"]
    evaluator: str
    selected_for_llm: bool = False
    overall_score: float = Field(default=0.0, ge=0.0, le=5.0)
    scores: dict[str, float] = Field(default_factory=dict)
    issues: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class RunManifest(StrictModel):
    run_id: str
    project_name: str
    status: Literal["planned", "running", "completed", "partial", "failed", "interrupted"]
    config_path: str
    plan_path: str
    database_path: str
    created_at: str = Field(default_factory=utc_now)
    updated_at: str = Field(default_factory=utc_now)
    environment: dict[str, Any] = Field(default_factory=dict)


class RunReport(StrictModel):
    run_id: str
    status: Literal["completed", "partial", "failed", "interrupted"]
    target_examples: int
    accepted_examples: int
    attempted_examples: int
    rejected_examples: int
    reviewed_examples: int
    speculative_examples: int = 0
    llm_judged_examples: int
    deficits: dict[str, int] = Field(default_factory=dict)
    metrics: dict[str, Any] = Field(default_factory=dict)
    finished_at: str = Field(default_factory=utc_now)
