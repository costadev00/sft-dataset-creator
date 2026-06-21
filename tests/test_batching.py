from __future__ import annotations

import json
import os
import sqlite3

import pytest
from pydantic import ValidationError

from sft_dataset_creator.backends import BackendProcess
from sft_dataset_creator.config import BatchingConfig, GenerationConfig, load_config
from sft_dataset_creator.engine import execute_plan
from sft_dataset_creator.models import ChatMessage, GenerationRequest
from sft_dataset_creator.planner import build_plan
from sft_dataset_creator.state import RunState
from sft_dataset_creator.tuning import tune_project


SCHEMA = {"type": "object", "properties": {"instruction": {"type": "string"}}}


def _request(request_id: str) -> GenerationRequest:
    return GenerationRequest(
        request_id=request_id,
        seed=7,
        slot_id=request_id,
        document_id="doc",
        task="closed_qa",
        difficulty="easy",
        messages=[ChatMessage(role="user", content="Return one grounded example.")],
        response_schema=SCHEMA,
        max_output_tokens=32,
    )


def test_batching_config_rejects_an_undersized_queue() -> None:
    with pytest.raises(ValidationError):
        BatchingConfig(max_inflight_requests=8, queue_capacity=4)


def test_async_backend_streams_completion_order() -> None:
    config = GenerationConfig(
        plugin="fake",
        model="fake",
        params={"delay_by_request": {"slow": 0.08, "fast": 0.0}},
        batching=BatchingConfig(mode="async", max_inflight_requests=2, queue_capacity=2),
    )
    with BackendProcess(config) as backend:
        results = list(backend.generate_many([_request("slow"), _request("fast")]))
    assert [result.request_id for result in results] == ["fast", "slow"]
    assert all(result.error is None for result in results)


def test_sequential_backend_preserves_submission_order() -> None:
    config = GenerationConfig(
        plugin="fake",
        model="fake",
        params={"delay_by_request": {"slow": 0.03, "fast": 0.0}},
        batching=BatchingConfig(mode="sequential", max_inflight_requests=2, queue_capacity=2),
    )
    with BackendProcess(config) as backend:
        results = list(backend.generate_many([_request("slow"), _request("fast")]))
    assert [result.request_id for result in results] == ["slow", "fast"]


def test_speculative_engine_commits_attempts_deterministically(project_config, tmp_path) -> None:
    delays = {
        f"slot-{slot:08d}-a1": 0.03
        for slot in range(1, project_config.target.examples + 1)
    }
    config = project_config.model_copy(
        update={
            "generation": project_config.generation.model_copy(
                update={"params": {"delay_by_request": delays}}
            )
        }
    )
    run_dir = tmp_path / "speculative"
    plan = build_plan(config, run_dir)
    report = execute_plan(plan, config, run_dir=run_dir)
    assert report.attempted_examples == 8
    assert report.speculative_examples == 4
    with RunState(run_dir / "run.db") as state:
        assert {candidate.attempt for candidate in state.accepted_candidates()} == {1}


def test_run_state_migrates_pre_batching_database(tmp_path) -> None:
    path = tmp_path / "old.db"
    connection = sqlite3.connect(path)
    connection.execute(
        "CREATE TABLE attempts (candidate_id TEXT PRIMARY KEY, slot_id TEXT, attempt_no INTEGER, "
        "document_id TEXT, status TEXT, candidate_json TEXT, request_json TEXT, response_json TEXT, "
        "evaluation_json TEXT, error TEXT, created_at TEXT)"
    )
    connection.close()
    with RunState(path) as state:
        columns = {row["name"] for row in state.connection.execute("PRAGMA table_info(attempts)")}
    assert {"request_id", "input_tokens", "output_tokens", "latency_seconds", "speculative"} <= columns


def test_tune_writes_resolved_config_and_report(project_config, tmp_path) -> None:
    output = tmp_path / "tuned.json"
    tuned, report_path = tune_project(project_config, output, stage="generation", samples=2)
    assert load_config(output) == tuned
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["resolved_config_hash"] == tuned.config_hash
    assert report["stages"]["generation"]["trials"][0]["name"] == "serial"


@pytest.mark.gpu
def test_cluster_profile_reaches_continuous_batching_targets(tmp_path) -> None:
    config_path = os.getenv("SFT_GPU_TEST_CONFIG")
    if not config_path:
        pytest.skip("set SFT_GPU_TEST_CONFIG to run the four-GPU benchmark")
    config = load_config(config_path)
    _tuned, report_path = tune_project(config, tmp_path / "gpu-tuned.json", stage="generation", samples=8)
    stage = json.loads(report_path.read_text(encoding="utf-8"))["stages"]["generation"]
    assert stage["meets_2x_target"]
    assert stage["meets_80_percent_gpu_target"]
