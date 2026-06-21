from __future__ import annotations

from typer.testing import CliRunner

from sft_dataset_creator.cli import app
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


def test_cli_run_with_fake_backend(project_config, tmp_path) -> None:
    config_path = save_config(project_config, tmp_path / "project.json")
    run_dir = tmp_path / "cli-run"
    result = runner.invoke(
        app,
        ["run", "--config", str(config_path), "--run-dir", str(run_dir)],
    )
    assert result.exit_code == 0, result.output
    assert (run_dir / "report.json").exists()
    assert (run_dir / "exports" / "alpaca" / "train.jsonl").exists()


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
