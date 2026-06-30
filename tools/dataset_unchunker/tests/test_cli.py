from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from typer.testing import CliRunner

from dataset_unchunker.cli import app


def write_fixture(path: Path) -> None:
    rows = [
        {
            "doc_id": "doc-1",
            "record_id": None,
            "title": "One",
            "text": "abcdef",
            "chunk_index": 0,
            "chunk_total": 2,
            "token_count": 6,
        },
        {
            "doc_id": "doc-1",
            "record_id": None,
            "title": "One",
            "text": "defghi",
            "chunk_index": 1,
            "chunk_total": 2,
            "token_count": 6,
        },
        {
            "doc_id": "doc-bad",
            "record_id": None,
            "title": "Bad",
            "text": "orphan",
            "chunk_index": 1,
            "chunk_total": 2,
            "token_count": 6,
        },
        {
            "doc_id": "doc-single",
            "record_id": "single-row-id",
            "title": "Single",
            "text": "complete",
            "chunk_index": 0,
            "chunk_total": 1,
            "token_count": 8,
        },
    ]
    pq.write_table(pa.Table.from_pylist(rows), path)


def test_cli_writes_parquet_report_and_quarantine(tmp_path) -> None:
    source = tmp_path / "chunks.parquet"
    output = tmp_path / "out"
    write_fixture(source)

    result = CliRunner().invoke(
        app,
        [
            "unchunk",
            "--source-file",
            str(source),
            "--text-column",
            "text",
            "--chunk-index-column",
            "chunk_index",
            "--chunk-total-column",
            "chunk_total",
            "--group-id-column",
            "doc_id",
            "--strategy",
            "text-overlap",
            "--max-overlap",
            "3",
            "--output-dir",
            str(output),
            "--no-push",
        ],
    )

    assert result.exit_code == 0, result.output
    table = pq.read_table(output / "data" / "train.parquet")
    rows = table.to_pylist()
    assert len(rows) == 2
    assert rows[0]["text"] == "abcdefghi"
    assert rows[0]["split"] == "train"
    assert "chunk_index" not in rows[0]
    assert "chunk_total" not in rows[0]
    assert "token_count" not in rows[0]
    assert rows[1]["record_id"] == "single-row-id"
    assert (output / "README.md").exists()
    report = (output / "reports" / "reconstruction_report.json").read_text(encoding="utf-8")
    assert '"reconstructed_groups": 2' in report
    quarantine = (output / "reports" / "quarantined_groups.jsonl").read_text(encoding="utf-8")
    assert "doc-bad" in quarantine
