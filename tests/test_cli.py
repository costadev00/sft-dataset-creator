from __future__ import annotations

import json
import os

from typer.testing import CliRunner

from sft_dataset_creator.cli import DEFAULT_MODEL, _configure_cache_environment, _model_params, app
from sft_dataset_creator.config import save_config


runner = CliRunner()


def test_cli_validate_and_schema(project_config, tmp_path) -> None:
    config_path = save_config(project_config, tmp_path / "project.json")
    result = runner.invoke(app, ["validate", "--config", str(config_path)])
    assert result.exit_code == 0, result.output
    schema_path = tmp_path / "schema.json"
    result = runner.invoke(app, ["schema", "--output", str(schema_path)])
    assert result.exit_code == 0, result.output
    assert schema_path.exists()


def test_cli_plugins() -> None:
    result = runner.invoke(app, ["plugins"])
    assert result.exit_code == 0, result.output
    assert "vllm_local" in result.output


def test_cli_run_help_has_no_judge_options() -> None:
    result = runner.invoke(app, ["run", "--help"])
    assert result.exit_code == 0, result.output
    for option in ("judge-model", "judge-plugin", "judge-param", "audit-fraction"):
        assert option not in result.output


def test_gemma_vllm_defaults_use_31b_qat_fp8_kv_cache(tmp_path) -> None:
    params = _model_params(
        [],
        "--generator-param",
        plugin="vllm_local",
        model=DEFAULT_MODEL,
        cache_dir=tmp_path,
    )
    assert DEFAULT_MODEL == "google/gemma-4-31B-it-qat-w4a16-ct"
    assert params["tensor_parallel_size"] == 4
    assert params["quantization"] == "compressed-tensors"
    assert params["kv_cache_dtype"] == "fp8"
    assert params["max_num_batched_tokens"] == 16_384
    assert params["download_dir"] == str((tmp_path / "models").resolve())


def test_explicit_gemma_vllm_batch_size_wins(tmp_path) -> None:
    params = _model_params(
        ["max_num_batched_tokens=8192"],
        "--generator-param",
        plugin="vllm_local",
        model=DEFAULT_MODEL,
        cache_dir=tmp_path,
    )
    assert params["max_num_batched_tokens"] == 8_192


def test_cli_cache_overrides_stale_huggingface_environment(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("HF_HOME", "/stale/unwritable/cache")
    _configure_cache_environment(tmp_path)
    assert os.environ["HF_HOME"] == str((tmp_path / "huggingface").resolve())
    assert os.environ["HF_HUB_CACHE"] == str((tmp_path / "huggingface" / "hub").resolve())
    assert os.environ["HF_XET_CACHE"] == str((tmp_path / "huggingface" / "xet").resolve())
    assert os.environ["TRANSFORMERS_CACHE"] == str(
        (tmp_path / "huggingface" / "transformers").resolve()
    )


def test_cli_run_with_fake_backend_without_input_config(corpus_path, tmp_path) -> None:
    run_dir = tmp_path / "cli-run"
    result = runner.invoke(
        app,
        [
            "run",
            "--dataset",
            str(corpus_path),
            "--source",
            "local",
            "--examples",
            "4",
            "--documents",
            "8",
            "--generator-plugin",
            "fake",
            "--model",
            "fake-generator",
            "--task",
            "closed_qa=1",
            "--difficulty",
            "easy=1",
            "--generator-param",
            "custom=true",
            "--smoke-models",
            "--run-dir",
            str(run_dir),
        ],
    )
    assert result.exit_code == 0, result.output
    assert (run_dir / "report.json").exists()
    assert (run_dir / "exports" / "alpaca" / "train.jsonl").exists()
    resolved = json.loads((run_dir / "config.resolved.json").read_text(encoding="utf-8"))
    assert resolved["source"]["params"]["path"] == str(corpus_path)
    assert resolved["target"]["examples"] == 4
    assert resolved["generation"]["params"]["custom"] is True
    assert resolved["evaluation"]["llm"] is None
    assert resolved["composition"]["tasks"]["weights"] == {"closed_qa": 1.0}
    resumed = runner.invoke(app, ["run", "--resume", str(run_dir)])
    assert resumed.exit_code == 0, resumed.output


def test_cli_run_requires_examples_with_dataset(corpus_path) -> None:
    result = runner.invoke(app, ["run", "--dataset", str(corpus_path), "--source", "local"])
    assert result.exit_code != 0
    assert "--examples is required with --dataset" in result.output


def test_cli_run_rejects_dataset_revision_placeholder(corpus_path) -> None:
    result = runner.invoke(
        app,
        [
            "run",
            "--dataset",
            str(corpus_path),
            "--source",
            "local",
            "--dataset-revision",
            "DATASET_COMMIT_SHA",
            "--examples",
            "2",
        ],
    )
    assert result.exit_code != 0
    assert "replace DATASET_COMMIT_SHA" in result.output


def test_cli_run_rejects_multiple_selection_sizes(corpus_path) -> None:
    result = runner.invoke(
        app,
        [
            "run",
            "--dataset",
            str(corpus_path),
            "--source",
            "local",
            "--examples",
            "2",
            "--documents",
            "2",
            "--selection-fraction",
            "0.5",
        ],
    )
    assert result.exit_code != 0
    assert "use only one of --documents or --selection-fraction" in result.output


def test_cli_run_rejects_unknown_task(corpus_path) -> None:
    result = runner.invoke(
        app,
        [
            "run",
            "--dataset",
            str(corpus_path),
            "--source",
            "local",
            "--examples",
            "2",
            "--task",
            "not-a-task=1",
        ],
    )
    assert result.exit_code != 0
    assert "unknown --task value(s): not-a-task" in result.output


def test_cli_tune_writes_profile_artifacts(project_config, tmp_path) -> None:
    config_path = save_config(project_config, tmp_path / "project.json")
    output = tmp_path / "project.tuned.json"
    result = runner.invoke(
        app,
        [
            "tune",
            "--config",
            str(config_path),
            "--output",
            str(output),
            "--stage",
            "generation",
            "--samples",
            "2",
        ],
    )
    assert result.exit_code == 0, result.output
    assert output.exists()
    assert (tmp_path / "project.tuned.tuning-report.json").exists()
