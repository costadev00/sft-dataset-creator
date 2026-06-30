from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any


class Strategy(StrEnum):
    AUTO = "auto"
    TOKEN_OVERLAP = "token-overlap"
    TEXT_OVERLAP = "text-overlap"
    CONCAT_NO_OVERLAP = "concat-no-overlap"


class SplitPolicy(StrEnum):
    TRAIN = "train"
    MAJORITY = "majority"
    ORIGINAL_IF_CONSISTENT = "original-if-consistent"
    HASH = "hash"


@dataclass(frozen=True)
class UnchunkConfig:
    text_column: str
    chunk_index_column: str
    chunk_total_column: str
    group_by: tuple[str, ...] = ()
    group_id_column: str | None = None
    tokenizer: str | None = None
    strategy: Strategy = Strategy.AUTO
    max_overlap: int = 256
    expected_overlap: int | None = None
    split_policy: SplitPolicy = SplitPolicy.TRAIN
    source_split_column: str = "split"
    drop_columns: tuple[str, ...] = ("token_count",)
    max_groups: int | None = None

    def validate(self) -> None:
        if bool(self.group_by) == bool(self.group_id_column):
            raise ValueError("provide exactly one of group_by or group_id_column")
        if self.max_overlap < 0:
            raise ValueError("max_overlap must be zero or positive")
        if self.expected_overlap is not None and self.expected_overlap < 0:
            raise ValueError("expected_overlap must be zero or positive")
        if self.strategy == Strategy.CONCAT_NO_OVERLAP and self.expected_overlap != 0:
            raise ValueError("concat-no-overlap requires expected_overlap=0")
        if self.strategy == Strategy.TOKEN_OVERLAP and not self.tokenizer:
            raise ValueError("token-overlap requires a tokenizer")

    @property
    def resolved_strategy(self) -> Strategy:
        if self.strategy != Strategy.AUTO:
            return self.strategy
        return Strategy.TOKEN_OVERLAP if self.tokenizer else Strategy.TEXT_OVERLAP

    @property
    def required_columns(self) -> set[str]:
        columns = {
            self.text_column,
            self.chunk_index_column,
            self.chunk_total_column,
        }
        columns.update(self.group_by)
        if self.group_id_column:
            columns.add(self.group_id_column)
        return columns

    @property
    def chunk_metadata_columns(self) -> set[str]:
        return {
            self.chunk_index_column,
            self.chunk_total_column,
            *self.drop_columns,
        }


@dataclass(frozen=True)
class SourceSpec:
    repo_id: str | None
    revision: str | None = None
    config_name: str | None = None
    splits: tuple[str, ...] = ("train",)
    source_file: str | None = None
    token_env: str = "HF_TOKEN"


@dataclass(frozen=True)
class OutputSpec:
    output_dir: Path
    output_repo_id: str | None = None
    private: bool = True
    no_push: bool = False
    token_env: str = "HF_TOKEN"


@dataclass
class QuarantineEntry:
    group_id: str
    reason: str
    details: dict[str, Any] = field(default_factory=dict)
    rows: int = 0


@dataclass
class ReconstructedGroup:
    group_id: str
    row: dict[str, Any]
    conflicts: dict[str, list[Any]]
    inferred_overlaps: list[int] = field(default_factory=list)
