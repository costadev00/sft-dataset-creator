from __future__ import annotations

import json

from sft_dataset_creator.checkpoints import CheckpointWriter
from sft_dataset_creator.config import DistributionConfig
from sft_dataset_creator.document_store import SQLiteDocumentStore
from sft_dataset_creator.engine import execute_plan
from sft_dataset_creator.planner import build_plan
from sft_dataset_creator.state import RunState


def test_checkpoint_writer_shards_and_resume_are_idempotent(project_config, tmp_path) -> None:
    config = project_config.model_copy(
        update={
            "runtime": project_config.runtime.model_copy(
                update={
                    "checkpoint": project_config.runtime.checkpoint.model_copy(
                        update={
                            "formats": ["canonical", "messages"],
                            "shard_size": 2,
                            "progress_interval_seconds": 0.01,
                        }
                    )
                }
            )
        }
    )
    run_dir = tmp_path / "checkpointed"
    plan = build_plan(config, run_dir)

    report = execute_plan(plan, config, run_dir=run_dir)

    assert report.status == "completed"
    canonical_parts = sorted((run_dir / "checkpoints" / "canonical" / "train").glob("*.jsonl"))
    assert [len(path.read_text(encoding="utf-8").splitlines()) for path in canonical_parts] == [2, 2]
    assert not list((run_dir / "checkpoints").glob("*/*/*.tmp"))
    with RunState(run_dir / "run.db") as state:
        assert len(state.checkpoint_shards()) == 4
        before = state.connection.execute("SELECT COUNT(*) FROM checkpoint_items").fetchone()[0]
        assert CheckpointWriter(run_dir, state, config).catch_up() == 0
        after = state.connection.execute("SELECT COUNT(*) FROM checkpoint_items").fetchone()[0]
    assert before == after


def test_document_store_recovers_documents_from_sqlite_shards(project_config, tmp_path) -> None:
    run_dir = tmp_path / "document-store"
    build_plan(project_config, run_dir)

    with RunState(run_dir / "run.db") as state:
        first_slot = state.slots(limit=1)[0]
        store = SQLiteDocumentStore(run_dir, state, cache_size=1)
        document = store.get(first_slot.document_id)

    assert document.id == first_slot.document_id
    assert document.sections
    assert "Document" in (document.title or "")


def test_progress_json_reflects_run_counts(project_config, tmp_path) -> None:
    run_dir = tmp_path / "progress"
    plan = build_plan(project_config, run_dir)
    execute_plan(plan, project_config, run_dir=run_dir)

    progress = json.loads((run_dir / "progress.json").read_text(encoding="utf-8"))

    assert progress["status"] == "completed"
    assert progress["target"] == 4
    assert progress["accepted"] == 4
    assert progress["accepted_percent"] == 100.0
    assert progress["eta_seconds"] is None
    assert "current_slot_id" not in progress
    assert progress["checkpoint_shards"]


def test_recent_rejections_are_dashboard_friendly(project_config, tmp_path) -> None:
    corpus = tmp_path / "short.jsonl"
    corpus.write_text(json.dumps({"id": "short", "title": "Short Article", "text": "tiny"}) + "\n", encoding="utf-8")
    config = project_config.model_copy(
        update={
            "source": project_config.source.model_copy(update={"params": {"path": str(corpus), "format": "jsonl"}}),
            "selection": project_config.selection.model_copy(update={"count": 1, "strata": []}),
            "target": project_config.target.model_copy(
                update={
                    "examples": 1,
                    "reserve_fraction": 0.0,
                    "max_attempts_per_slot": 1,
                    "max_total_attempt_multiplier": 1.0,
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
    run_dir = tmp_path / "dashboard-rejections"
    plan = build_plan(config, run_dir)
    execute_plan(plan, config, run_dir=run_dir)

    with RunState(run_dir / "run.db") as state:
        items = state.recent_rejections()

    assert items
    assert items[0]["title"] == "Short Article"
    assert items[0]["reason"]
    assert "evaluation_json" not in items[0]


def test_recent_successes_are_dashboard_friendly(project_config, tmp_path) -> None:
    run_dir = tmp_path / "dashboard-successes"
    plan = build_plan(project_config, run_dir)
    execute_plan(plan, project_config, run_dir=run_dir)

    with RunState(run_dir / "run.db") as state:
        items = state.recent_successes()

    assert items
    assert items[0]["title"]
    assert items[0]["question"]
    assert items[0]["answer"]
    assert "candidate_json" not in items[0]
