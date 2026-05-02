from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from wiki_if_builder.config import DEFAULT_DATASET_NAME, DEFAULT_LICENSE


class StrictBaseModel(BaseModel):
    model_config = ConfigDict(extra="ignore")


class SourceMetadata(StrictBaseModel):
    dataset: str = DEFAULT_DATASET_NAME
    page_id: int
    title: str
    license: str = DEFAULT_LICENSE


class TriageResult(StrictBaseModel):
    status: str
    quality_flags: list[str] = Field(default_factory=list)
    char_count: int = 0
    word_count: int = 0
    unique_word_count: int = 0
    alpha_ratio: float = 0.0
    symbolic_ratio: float = 0.0
    reason: str = ""
    requires_section_fallback: bool = False
    context_truncated: bool = False
    used_char_count: int = 0
    original_char_count: int = 0


class DocumentLabels(StrictBaseModel):
    primary_category: str = "outros"
    subcategory: str = "unknown"
    secondary_categories: list[str] = Field(default_factory=list)
    document_type: str = "unknown"
    task_affordances: list[str] = Field(default_factory=list)
    if_eligibility: bool = False
    difficulty_band: str = "medium"
    grounding_profile: str = "unknown"
    risk_band: str = "medium"


class EvidenceRef(StrictBaseModel):
    section_id: int = 0
    char_start: int = 0
    char_end: int = 0

    @model_validator(mode="after")
    def _validate_offsets(self) -> "EvidenceRef":
        self.char_start = max(0, self.char_start)
        self.char_end = max(self.char_start, self.char_end)
        return self


class IFCandidate(StrictBaseModel):
    candidate_id: str = ""
    task_type: str = "closed_qa"
    instruction: str = ""
    input: str = ""
    completion: str = ""
    style: str = "didatico"
    difficulty: str = "medium"
    source_refs: list[EvidenceRef] = Field(default_factory=list)


class AnalystOutput(StrictBaseModel):
    record_id: str
    source: SourceMetadata
    triage: TriageResult
    document_labels: DocumentLabels
    evidence_refs: list[EvidenceRef] = Field(default_factory=list)
    if_candidates: list[IFCandidate] = Field(default_factory=list)
    model_name: str | None = None
    pipeline_version: str | None = None


class JudgeScores(StrictBaseModel):
    grounding: float = 0.0
    instruction_quality: float = 0.0
    completion_quality: float = 0.0
    non_triviality: float = 0.0
    style_clarity: float = 0.0
    schema_validity: float = 0.0

    @field_validator(
        "grounding",
        "instruction_quality",
        "completion_quality",
        "non_triviality",
        "style_clarity",
        "schema_validity",
    )
    @classmethod
    def _score_range(cls, value: float) -> float:
        return max(0.0, min(5.0, float(value)))


class JudgeDecision(StrictBaseModel):
    verdict: Literal["accept", "reject", "review"] = "review"
    overall_score: float = 0.0
    scores: JudgeScores = Field(default_factory=JudgeScores)
    issues: list[str] = Field(default_factory=list)
    needs_human_review: bool = True

    @field_validator("overall_score")
    @classmethod
    def _overall_range(cls, value: float) -> float:
        return max(0.0, min(5.0, float(value)))


class JudgeResult(StrictBaseModel):
    candidate_id: str
    judge: JudgeDecision = Field(default_factory=JudgeDecision)
    source_page_id: int | None = None
    source_title: str | None = None
    judge_model_name: str | None = None


class DocumentLabelsDatasetRecord(StrictBaseModel):
    id: str
    source_dataset: str
    page_id: int
    title: str
    license: str
    triage_status: str
    quality_flags: list[str]
    char_count: int
    word_count: int
    primary_category: str
    subcategory: str
    secondary_categories: list[str]
    document_type: str
    task_affordances: list[str]
    if_eligibility: bool
    difficulty_band: str
    grounding_profile: str
    risk_band: str
    requires_section_fallback: bool
    context_truncated: bool
    used_char_count: int
    original_char_count: int
    model_name: str
    pipeline_version: str


class MessageRecord(StrictBaseModel):
    role: Literal["user", "assistant", "system"]
    content: str


class InstructionFollowingDatasetRecord(StrictBaseModel):
    id: str
    source_dataset: str
    source_page_id: int
    source_title: str
    license: str
    instruction: str
    input: str
    output: str
    messages: list[MessageRecord]
    task_type: str
    primary_category: str
    subcategory: str
    difficulty: str
    style: str
    judge_verdict: str
    judge_score: float
    judge_grounding: float
    judge_instruction_quality: float
    judge_completion_quality: float
    judge_non_triviality: float
    requires_section_fallback: bool
    context_truncated: bool
    model_name: str
    judge_model_name: str
    pipeline_version: str


class SFTPromptCompletionRecord(StrictBaseModel):
    id: str
    prompt: str
    completion: str
    source_page_id: int
    source_title: str
    license: str = DEFAULT_LICENSE


class SFTMessagesRecord(StrictBaseModel):
    id: str
    messages: list[MessageRecord]
    source_page_id: int
    source_title: str
    license: str = DEFAULT_LICENSE


class PreferencePairRecord(StrictBaseModel):
    id: str
    prompt: str
    chosen: str
    rejected: str
    source_page_id: int
    source_title: str
    license: str = DEFAULT_LICENSE

