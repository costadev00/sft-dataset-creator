from __future__ import annotations

import json

import pytest

from sft_dataset_creator.config import (
    DistributionConfig,
    EvaluationConfig,
    GenerationConfig,
    PerDocumentConfig,
    SourceConfig,
)
from sft_dataset_creator.engine import execute_plan
from sft_dataset_creator.exporters import export_run
from sft_dataset_creator.models import EvaluationResult
from sft_dataset_creator.planner import build_plan
from sft_dataset_creator.state import RunState


def test_fake_backend_executes_and_resumes(project_config, tmp_path) -> None:
    run_dir = tmp_path / "run"
    plan = build_plan(project_config, run_dir)
    report = execute_plan(plan, project_config, run_dir=run_dir)
    assert report.status == "completed"
    assert report.accepted_examples == 4
    assert (run_dir / "exports" / "messages" / "train.jsonl").exists()

    second = execute_plan(plan, project_config, run_dir=run_dir)
    assert second.attempted_examples == report.attempted_examples
    with RunState(run_dir / "run.db") as state:
        assert len(list(state.accepted_candidates())) == 4


def test_final_exports_do_not_leak_documents_between_splits(project_config, tmp_path) -> None:
    run_dir = tmp_path / "run"
    plan = build_plan(project_config, run_dir)
    execute_plan(plan, project_config, run_dir=run_dir)
    seen = {}
    for split in ("train", "validation", "test"):
        path = run_dir / "exports" / "messages" / f"{split}.jsonl"
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            assert row["document_id"] not in seen or seen[row["document_id"]] == split
            seen[row["document_id"]] = split


def test_export_with_train_only_omits_disabled_splits(project_config, tmp_path) -> None:
    project_config.output.splits.train = 1.0
    project_config.output.splits.validation = 0.0
    project_config.output.splits.test = 0.0
    run_dir = tmp_path / "train-only"
    plan = build_plan(project_config, run_dir)
    execute_plan(plan, project_config, run_dir=run_dir)

    format_dir = run_dir / "exports" / "messages"
    assert (format_dir / "train.jsonl").is_file()
    assert not (format_dir / "validation.jsonl").exists()
    assert not (format_dir / "test.jsonl").exists()


def test_export_refuses_source_dependent_accepted_candidate(project_config, tmp_path) -> None:
    run_dir = tmp_path / "source-dependent"
    plan = build_plan(project_config, run_dir)
    execute_plan(plan, project_config, run_dir=run_dir)
    with RunState(run_dir / "run.db") as state:
        candidate = next(state.accepted_candidates()).model_copy(
            update={"instruction": "De acordo com o texto, qual é a resposta?"}
        )
        state.record_evaluation(
            candidate,
            EvaluationResult(
                candidate_id=candidate.id,
                verdict="accept",
                evaluator="legacy",
            ),
        )

    with pytest.raises(ValueError, match="source-dependent SFT content"):
        export_run(run_dir, config=project_config)


def test_multiple_examples_from_document_iterate_over_chunks(project_config, tmp_path) -> None:
    corpus = tmp_path / "long.jsonl"
    corpus.write_text(json.dumps({"id": "long", "text": "A" * 1_300}) + "\n", encoding="utf-8")
    config = project_config.model_copy(
        update={
            "source": SourceConfig(
                plugin="local",
                params={"path": str(corpus), "format": "jsonl"},
                field_map={"id": "id", "text": "text"},
            ),
            "selection": project_config.selection.model_copy(update={"count": 1, "strata": []}),
            "target": project_config.target.model_copy(
                update={
                    "examples": 3,
                    "per_document": PerDocumentConfig(minimum=3, maximum=3),
                    "reserve_fraction": 0.0,
                    "max_attempts_per_slot": 1,
                    "max_total_attempt_multiplier": 1.0,
                    "chunk_size_characters": 500,
                    "chunk_overlap_characters": 0,
                }
            ),
        }
    )
    run_dir = tmp_path / "iterative-chunks"
    plan = build_plan(config, run_dir)

    report = execute_plan(plan, config, run_dir=run_dir)

    assert report.accepted_examples == 3
    assert [slot.chunk_id for slot in plan.slots] == ["0", "1", "2"]
    with RunState(run_dir / "run.db") as state:
        assert [candidate.evidence[0].section_id for candidate in state.accepted_candidates()] == ["0", "1", "2"]


def test_selective_llm_evaluation_uses_second_backend_process(project_config, tmp_path) -> None:
    config = project_config.model_copy(
        update={
            "composition": project_config.composition.model_copy(
                update={"difficulties": DistributionConfig(counts={"hard": 4})}
            ),
            "evaluation": EvaluationConfig(
                llm=GenerationConfig(plugin="fake", model="fake-judge")
            ),
        }
    )
    run_dir = tmp_path / "hybrid"
    plan = build_plan(config, run_dir)
    report = execute_plan(plan, config, run_dir=run_dir)
    assert report.status == "completed"
    assert report.llm_judged_examples == 4


def test_attempt_limits_produce_explicit_partial_report(project_config, tmp_path) -> None:
    corpus = tmp_path / "short.jsonl"
    corpus.write_text(json.dumps({"id": "short", "text": "tiny"}) + "\n", encoding="utf-8")
    config = project_config.model_copy(
        update={
            "source": project_config.source.model_copy(update={"params": {"path": str(corpus), "format": "jsonl"}}),
            "selection": project_config.selection.model_copy(update={"count": 1, "strata": []}),
            "target": project_config.target.model_copy(
                update={
                    "examples": 1,
                    "reserve_fraction": 0.25,
                    "max_attempts_per_slot": 2,
                    "max_total_attempt_multiplier": 2.0,
                }
            ),
            "composition": project_config.composition.model_copy(
                update={
                    "tasks": DistributionConfig(counts={"closed_qa": 1}),
                    "difficulties": DistributionConfig(counts={"easy": 1}),
                }
            ),
        }
    )
    run_dir = tmp_path / "partial"
    plan = build_plan(config, run_dir)
    report = execute_plan(plan, config, run_dir=run_dir)
    assert report.status == "partial"
    assert report.accepted_examples == 0
    assert report.attempted_examples == 2
    assert report.deficits == {"closed_qa:easy": 1}


def test_parquet_export(project_config, tmp_path) -> None:
    config = project_config.model_copy(
        update={
            "output": project_config.output.model_copy(update={"containers": ["jsonl", "parquet"]})
        }
    )
    run_dir = tmp_path / "parquet"
    plan = build_plan(config, run_dir)
    execute_plan(plan, config, run_dir=run_dir)
    assert (run_dir / "exports" / "messages" / "train.parquet").exists()
