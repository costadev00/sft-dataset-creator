from __future__ import annotations

from collections import Counter
from pathlib import Path

from sft_dataset_creator.config import PerDocumentConfig
from sft_dataset_creator.planner import build_plan, load_document_snapshot, load_plan


def test_planner_builds_exact_deterministic_quotas(project_config, tmp_path) -> None:
    plan = build_plan(project_config, tmp_path / "run-a")
    assert len(plan.slots) == 4
    assert len(plan.reserve_document_ids) == 2
    assert Counter(slot.task for slot in plan.slots) == {"closed_qa": 2, "summarization": 2}
    assert Counter(slot.difficulty for slot in plan.slots) == {"easy": 2, "medium": 2}
    assert Path(plan.corpus_snapshot).exists()
    assert len(load_document_snapshot(plan.corpus_snapshot)) == 8
    assert load_plan(tmp_path / "run-a" / "plan.json") == plan


def test_planner_rejects_insufficient_primary_capacity(project_config, tmp_path) -> None:
    config = project_config.model_copy(
        update={
            "selection": project_config.selection.model_copy(update={"count": 4}),
            "target": project_config.target.model_copy(
                update={"examples": 10, "per_document": PerDocumentConfig(minimum=1, maximum=2)}
            ),
        }
    )
    import pytest

    with pytest.raises(ValueError, match="cannot satisfy target"):
        build_plan(config, tmp_path / "insufficient")


def test_planner_respects_per_document_minimum(project_config, tmp_path) -> None:
    config = project_config.model_copy(
        update={
            "target": project_config.target.model_copy(
                update={"examples": 4, "per_document": PerDocumentConfig(minimum=2, maximum=3)}
            )
        }
    )
    plan = build_plan(config, tmp_path / "minimum")
    assert sorted(Counter(slot.document_id for slot in plan.slots).values()) == [2, 2]
