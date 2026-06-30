from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

from sft_dataset_creator.config import SourceConfig
from sft_dataset_creator.profiles import document_is_eligible
from sft_dataset_creator.registry import create


@dataclass(frozen=True)
class PreflightResult:
    total_rows: int
    eligible_documents: int
    selected_documents: int
    per_document_minimum: int
    per_document_maximum: int

    @property
    def minimum_examples(self) -> int:
        return self.selected_documents * self.per_document_minimum

    @property
    def maximum_examples(self) -> int:
        return self.selected_documents * self.per_document_maximum

    @property
    def recommended_examples(self) -> int | None:
        if self.per_document_minimum != self.per_document_maximum:
            return None
        return self.selected_documents * self.per_document_maximum


def count_preflight(
    source_config: SourceConfig,
    *,
    profile: str | None,
    document_count: int | None,
    selection_fraction: float | None,
    per_document_minimum: int,
    per_document_maximum: int,
    progress_every: int = 100_000,
    progress_callback: Callable[[int, int], None] | None = None,
) -> PreflightResult:
    if document_count is not None and selection_fraction is not None:
        raise ValueError("use only one of document_count or selection_fraction")
    if selection_fraction is not None and selection_fraction <= 0.0:
        raise ValueError("selection_fraction must be greater than zero")
    if per_document_minimum > per_document_maximum:
        raise ValueError("per-document minimum cannot exceed maximum")

    source = create("sources", source_config.plugin, source_config)
    total = 0
    eligible = 0
    for document in source.iter_documents():
        total += 1
        if document_is_eligible(document, profile):
            eligible += 1
        if progress_callback is not None and progress_every > 0 and total % progress_every == 0:
            progress_callback(total, eligible)

    if document_count is not None:
        selected = min(document_count, eligible)
    else:
        fraction = selection_fraction if selection_fraction is not None else 1.0
        selected = max(1, math.ceil(eligible * fraction)) if eligible else 0

    return PreflightResult(
        total_rows=total,
        eligible_documents=eligible,
        selected_documents=selected,
        per_document_minimum=per_document_minimum,
        per_document_maximum=per_document_maximum,
    )
