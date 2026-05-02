import orjson
from typer.testing import CliRunner

from wiki_if_builder.cli import app
from wiki_if_builder.exporter import (
    export_document_labels_dataset,
    export_instruction_following_dataset,
)
from wiki_if_builder.schemas import AnalystOutput, JudgeResult
from wiki_if_builder.utils import iter_jsonl


def _normalized_output():
    return AnalystOutput.model_validate(
        {
            "record_id": "wiki-ptbr-123",
            "source": {
                "dataset": "costadev00/wikipedia-pt-br-extract",
                "page_id": 123,
                "title": "Astronomia",
                "license": "cc-by-sa-3.0",
            },
            "triage": {
                "status": "valid_article",
                "quality_flags": [],
                "char_count": 1200,
                "word_count": 160,
                "unique_word_count": 90,
                "alpha_ratio": 0.8,
                "symbolic_ratio": 0.04,
                "reason": "ok",
                "requires_section_fallback": False,
                "context_truncated": False,
                "used_char_count": 1200,
                "original_char_count": 1200,
            },
            "document_labels": {
                "primary_category": "ciencia",
                "subcategory": "astronomia",
                "secondary_categories": ["fisica"],
                "document_type": "encyclopedic_article",
                "task_affordances": ["definition"],
                "if_eligibility": True,
                "difficulty_band": "medium",
                "grounding_profile": "strong",
                "risk_band": "low",
            },
            "evidence_refs": [{"section_id": 0, "char_start": 0, "char_end": 500}],
            "if_candidates": [
                {
                    "candidate_id": "wiki-ptbr-123-c1",
                    "task_type": "definition",
                    "instruction": "Explique o que é astronomia.",
                    "input": "Contexto mínimo.",
                    "completion": "Astronomia é a ciência que estuda corpos celestes.",
                    "style": "didatico",
                    "difficulty": "easy",
                    "source_refs": [{"section_id": 0, "char_start": 0, "char_end": 500}],
                }
            ],
            "model_name": "gemma-local",
            "pipeline_version": "0.1.0",
        }
    )


def _judge_result():
    return JudgeResult.model_validate(
        {
            "candidate_id": "wiki-ptbr-123-c1",
            "source_page_id": 123,
            "source_title": "Astronomia",
            "judge_model_name": "gemma-local",
            "judge": {
                "verdict": "accept",
                "overall_score": 4.7,
                "scores": {
                    "grounding": 5,
                    "instruction_quality": 4,
                    "completion_quality": 5,
                    "non_triviality": 4,
                    "style_clarity": 5,
                    "schema_validity": 5,
                },
                "issues": [],
                "needs_human_review": False,
            },
        }
    )


def test_exporter_writes_valid_jsonl(tmp_path):
    out = tmp_path / "outputs"
    labels_dir = export_document_labels_dataset([_normalized_output()], out)
    rows = list(iter_jsonl(labels_dir / "data.jsonl"))
    assert rows[0]["page_id"] == 123
    assert orjson.dumps(rows[0])


def test_exporter_creates_document_labels_data_jsonl(tmp_path):
    labels_dir = export_document_labels_dataset([_normalized_output()], tmp_path / "outputs")
    assert (labels_dir / "data.jsonl").exists()
    assert (labels_dir / "README.md").exists()
    assert (labels_dir / "dataset_info.json").exists()


def test_exporter_creates_instruction_following_data_jsonl(tmp_path):
    if_dir = export_instruction_following_dataset(
        [_normalized_output()],
        [_judge_result()],
        tmp_path / "outputs",
    )
    rows = list(iter_jsonl(if_dir / "data.jsonl"))
    assert (if_dir / "data.jsonl").exists()
    assert rows[0]["id"] == "wiki-ptbr-123-c1"
    assert rows[0]["messages"][0]["role"] == "user"


def test_dry_run_executes_without_llm(monkeypatch, tmp_path):
    def fail_if_called(*args, **kwargs):
        raise AssertionError("LLM client não deve ser criado no dry-run")

    monkeypatch.setattr("wiki_if_builder.cli.build_llm_client", fail_if_called)
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "run",
            "--max-articles",
            "2",
            "--dry-run",
            "true",
            "--output-dir",
            str(tmp_path / "outputs"),
            "--cache-dir",
            str(tmp_path / "cache"),
            "--tmp-dir",
            str(tmp_path / "tmp"),
            "--min-free-output-gb",
            "0",
            "--min-free-cache-gb",
            "0",
        ],
    )
    assert result.exit_code == 0, result.output
    assert (tmp_path / "outputs" / "document_labels" / "data.jsonl").exists()
    assert (tmp_path / "outputs" / "instruction_following" / "data.jsonl").exists()

