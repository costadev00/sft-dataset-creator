from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class ConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ComponentConfig(ConfigModel):
    plugin: str
    params: dict[str, Any] = Field(default_factory=dict)


class SourceConfig(ComponentConfig):
    field_map: dict[str, str] = Field(default_factory=dict)


class CorpusSelection(ConfigModel):
    count: int | None = Field(default=None, gt=0)
    fraction: float | None = Field(default=None, gt=0.0, le=1.0)
    seed: int = 42
    strata: list[str] = Field(default_factory=list)
    filters: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def exactly_one_size(self) -> "CorpusSelection":
        if (self.count is None) == (self.fraction is None):
            raise ValueError("selection must define exactly one of count or fraction")
        return self


class DistributionConfig(ConfigModel):
    counts: dict[str, int] | None = None
    weights: dict[str, float] | None = None

    @model_validator(mode="after")
    def exactly_one_distribution(self) -> "DistributionConfig":
        if (self.counts is None) == (self.weights is None):
            raise ValueError("distribution must define exactly one of counts or weights")
        values = self.counts if self.counts is not None else self.weights
        if not values or any(value <= 0 for value in values.values()):
            raise ValueError("distribution values must be positive")
        return self


class PerDocumentConfig(ConfigModel):
    minimum: int = Field(default=1, ge=0)
    maximum: int = Field(default=3, ge=1)

    @model_validator(mode="after")
    def validate_range(self) -> "PerDocumentConfig":
        if self.minimum > self.maximum:
            raise ValueError("per-document minimum cannot exceed maximum")
        return self


class TargetConfig(ConfigModel):
    examples: int = Field(gt=0)
    per_document: PerDocumentConfig = Field(default_factory=PerDocumentConfig)
    reserve_fraction: float = Field(default=0.25, ge=0.0, lt=1.0)
    same_document_attempts: int = Field(default=2, ge=1)
    max_attempts_per_slot: int = Field(default=3, ge=1)
    max_total_attempt_multiplier: float = Field(default=2.0, ge=1.0)
    chunk_size_characters: int = Field(default=8_000, ge=500)
    chunk_overlap_characters: int = Field(default=400, ge=0)

    @model_validator(mode="after")
    def validate_chunking(self) -> "TargetConfig":
        if self.chunk_overlap_characters >= self.chunk_size_characters:
            raise ValueError("chunk overlap must be smaller than chunk size")
        return self


class CompositionConfig(ConfigModel):
    tasks: DistributionConfig = Field(
        default_factory=lambda: DistributionConfig(
            weights={
                "closed_qa": 0.25,
                "summarization": 0.20,
                "information_extraction": 0.20,
                "concept_explanation": 0.20,
                "classification": 0.15,
            }
        )
    )
    difficulties: DistributionConfig = Field(
        default_factory=lambda: DistributionConfig(weights={"easy": 0.25, "medium": 0.50, "hard": 0.25})
    )
    grounding_required: bool = True


class BatchingConfig(ConfigModel):
    mode: Literal["auto", "async", "sequential"] = "auto"
    max_inflight_requests: int = Field(default=32, ge=1)
    queue_capacity: int = Field(default=64, ge=1)
    preparation_batch_size: int = Field(default=128, ge=1)
    request_timeout_seconds: float = Field(default=900.0, gt=0.0)

    @model_validator(mode="after")
    def validate_queue_capacity(self) -> "BatchingConfig":
        if self.queue_capacity < self.max_inflight_requests:
            raise ValueError("batching queue_capacity cannot be smaller than max_inflight_requests")
        return self


class GenerationConfig(ComponentConfig):
    model: str
    model_revision: str | None = None
    context_window: int = Field(default=65_536, gt=0)
    max_input_tokens: int = Field(default=53_248, gt=0)
    max_output_tokens: int = Field(default=4_096, gt=0)
    temperature: float = Field(default=0.1, ge=0.0)
    batching: BatchingConfig = Field(default_factory=BatchingConfig)

    @model_validator(mode="after")
    def validate_context_budget(self) -> "GenerationConfig":
        if self.max_input_tokens + self.max_output_tokens > self.context_window:
            raise ValueError("input and output token budgets exceed context window")
        return self


class RoutingConfig(ConfigModel):
    audit_fraction: float = Field(default=0.10, ge=0.0, le=1.0)
    judge_hard_tasks: bool = True
    judge_truncated_context: bool = True
    judge_high_risk: bool = True
    judge_weak_grounding: bool = True


class AcceptanceConfig(ConfigModel):
    overall_score: float = Field(default=4.0, ge=0.0, le=5.0)
    grounding_score: float = Field(default=4.0, ge=0.0, le=5.0)


class EvaluationConfig(ConfigModel):
    plugin: str = "composite"
    deterministic: bool = True
    llm: GenerationConfig | None = None
    routing: RoutingConfig = Field(default_factory=RoutingConfig)
    acceptance: AcceptanceConfig = Field(default_factory=AcceptanceConfig)
    # Retained only so resolved configurations from pre-0.4 runs keep their hashes.
    near_duplicate_threshold: float = Field(default=0.92, gt=0.0, le=1.0)


class SplitConfig(ConfigModel):
    train: float = Field(default=0.90, ge=0.0, le=1.0)
    validation: float = Field(default=0.05, ge=0.0, le=1.0)
    test: float = Field(default=0.05, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def sum_to_one(self) -> "SplitConfig":
        if abs(self.train + self.validation + self.test - 1.0) > 1e-6:
            raise ValueError("split ratios must sum to 1")
        return self


class OutputConfig(ConfigModel):
    formats: list[Literal["messages", "prompt_completion", "alpaca"]] = Field(
        default_factory=lambda: ["messages", "prompt_completion", "alpaca"]
    )
    containers: list[Literal["jsonl", "parquet"]] = Field(default_factory=lambda: ["jsonl"])
    splits: SplitConfig = Field(default_factory=SplitConfig)


class RuntimeConfig(ConfigModel):
    run_root: Path = Path("runs")
    cache_dir: Path = Path(".cache/sft-dataset-creator")
    store_model_io: bool = True
    fail_on_partial: bool = True

    @field_validator("run_root", "cache_dir", mode="before")
    @classmethod
    def expand_path(cls, value: Any) -> Path:
        return Path(value).expanduser()


class PublishConfig(ConfigModel):
    repo_id: str | None = None
    private: bool = True
    token_env: str = "HF_TOKEN"


class ProjectConfig(ConfigModel):
    name: str
    language: str = "en"
    profile: str | None = None
    source: SourceConfig
    selection: CorpusSelection
    target: TargetConfig
    composition: CompositionConfig = Field(default_factory=CompositionConfig)
    generation: GenerationConfig
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    publish: PublishConfig = Field(default_factory=PublishConfig)

    def canonical_json(self) -> str:
        payload = self.model_dump(mode="json")
        target = payload["target"]
        if self.target.chunk_size_characters == 8_000:
            target.pop("chunk_size_characters", None)
        if self.target.chunk_overlap_characters == 400:
            target.pop("chunk_overlap_characters", None)
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))

    @property
    def config_hash(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


def load_config(path: str | Path) -> ProjectConfig:
    config_path = Path(path)
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON configuration at {config_path}: {exc}") from exc
    return ProjectConfig.model_validate(payload)


def save_config(config: ProjectConfig, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(config.model_dump_json(indent=2), encoding="utf-8")
    return target


def write_config_schema(path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(ProjectConfig.model_json_schema(), indent=2), encoding="utf-8")
    return target


def gemma_cluster_preset(
    *,
    name: str,
    source: SourceConfig,
    selection: CorpusSelection,
    examples: int,
    language: str = "en",
    profile: str | None = None,
) -> ProjectConfig:
    return ProjectConfig(
        name=name,
        language=language,
        profile=profile,
        source=source,
        selection=selection,
        target=TargetConfig(examples=examples),
        generation=GenerationConfig(
            plugin="vllm_local",
            model="google/gemma-4-26B-A4B-it",
            batching=BatchingConfig(mode="async"),
            params={
                "tensor_parallel_size": 4,
                "gpu_memory_utilization": 0.92,
                "dtype": "bfloat16",
                "enable_thinking": False,
                "max_num_seqs": 16,
                "max_num_batched_tokens": 16_384,
                "enable_chunked_prefill": True,
                "enable_prefix_caching": True,
            },
        ),
        evaluation=EvaluationConfig(
            llm=GenerationConfig(
                plugin="vllm_local",
                model="google/gemma-4-31B-it-qat-w4a16-ct",
                batching=BatchingConfig(mode="async"),
                params={
                    "tensor_parallel_size": 4,
                    "gpu_memory_utilization": 0.92,
                    "quantization": "compressed-tensors",
                    "enable_thinking": True,
                    "max_num_seqs": 16,
                    "max_num_batched_tokens": 16_384,
                    "enable_chunked_prefill": True,
                    "enable_prefix_caching": True,
                },
            )
        ),
    )
