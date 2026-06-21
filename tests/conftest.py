from __future__ import annotations

import json
from pathlib import Path

import pytest

from sft_dataset_creator.config import (
    CompositionConfig,
    CorpusSelection,
    DistributionConfig,
    GenerationConfig,
    ProjectConfig,
    RuntimeConfig,
    SourceConfig,
    TargetConfig,
)


@pytest.fixture
def corpus_path(tmp_path: Path) -> Path:
    path = tmp_path / "corpus.jsonl"
    records = [
        {
            "id": f"doc-{index}",
            "title": f"Document {index}",
            "text": (
                f"Document {index} contains a grounded factual passage for synthetic dataset testing. "
                "It has enough text to validate evidence spans and produce a useful supervised example."
            ),
            "domain": "science" if index % 2 == 0 else "history",
        }
        for index in range(8)
    ]
    path.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")
    return path


@pytest.fixture
def project_config(tmp_path: Path, corpus_path: Path) -> ProjectConfig:
    return ProjectConfig(
        name="test-project",
        source=SourceConfig(
            plugin="local",
            params={"path": str(corpus_path), "format": "jsonl"},
            field_map={"id": "id", "text": "text", "title": "title"},
        ),
        selection=CorpusSelection(count=8, seed=7, strata=["metadata.domain"]),
        target=TargetConfig(examples=4, reserve_fraction=0.25),
        composition=CompositionConfig(
            tasks=DistributionConfig(weights={"closed_qa": 0.5, "summarization": 0.5}),
            difficulties=DistributionConfig(weights={"easy": 0.5, "medium": 0.5}),
        ),
        generation=GenerationConfig(plugin="fake", model="fake-generator"),
        runtime=RuntimeConfig(run_root=tmp_path / "runs", cache_dir=tmp_path / "cache"),
    )
