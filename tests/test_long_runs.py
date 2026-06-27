from __future__ import annotations

import json

from sft_dataset_creator.checkpoints import CheckpointWriter
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
    assert progress["checkpoint_shards"]
