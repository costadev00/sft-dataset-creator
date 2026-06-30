from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from dataset_unchunker.schema import QuarantineEntry, SourceSpec, UnchunkConfig


@dataclass
class ReconstructionReport:
    source: dict[str, Any]
    config: dict[str, Any]
    input_rows: int
    groups_seen: int
    reconstructed_groups: int
    quarantined_groups: int
    conflicts_by_column: dict[str, int] = field(default_factory=dict)
    overlap_histogram: dict[str, int] = field(default_factory=dict)


def make_report(
    *,
    source: SourceSpec,
    config: UnchunkConfig,
    input_rows: int,
    groups_seen: int,
    reconstructed_groups: int,
    quarantined_groups: int,
    conflict_counter: Counter[str],
    overlap_counter: Counter[int],
) -> ReconstructionReport:
    return ReconstructionReport(
        source={
            "repo_id": source.repo_id,
            "revision": source.revision,
            "config": source.config_name,
            "splits": list(source.splits),
            "source_file": source.source_file,
        },
        config={
            "text_column": config.text_column,
            "chunk_index_column": config.chunk_index_column,
            "chunk_total_column": config.chunk_total_column,
            "group_by": list(config.group_by),
            "group_id_column": config.group_id_column,
            "strategy": str(config.resolved_strategy),
            "tokenizer": config.tokenizer,
            "max_overlap": config.max_overlap,
            "expected_overlap": config.expected_overlap,
            "split_policy": str(config.split_policy),
            "drop_columns": list(config.drop_columns),
        },
        input_rows=input_rows,
        groups_seen=groups_seen,
        reconstructed_groups=reconstructed_groups,
        quarantined_groups=quarantined_groups,
        conflicts_by_column=dict(sorted(conflict_counter.items())),
        overlap_histogram={str(key): value for key, value in sorted(overlap_counter.items())},
    )


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(value) if hasattr(value, "__dataclass_fields__") else value
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_quarantine(path: Path, entries: list[QuarantineEntry]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(asdict(entry), sort_keys=True) + "\n")


def dataset_card(report: ReconstructionReport) -> str:
    source_repo = report.source.get("repo_id") or "local parquet"
    revision = report.source.get("revision") or "not pinned"
    return (
        "---\n"
        "task_categories:\n"
        "- text-generation\n"
        "format:\n"
        "- parquet\n"
        "---\n\n"
        "# Reconstructed Dataset\n\n"
        "This dataset was reconstructed from chunk rows by `dataset-unchunker`.\n\n"
        f"- Source: `{source_repo}`\n"
        f"- Source revision: `{revision}`\n"
        f"- Strategy: `{report.config['strategy']}`\n"
        f"- Input rows: {report.input_rows}\n"
        f"- Reconstructed groups: {report.reconstructed_groups}\n"
        f"- Quarantined groups: {report.quarantined_groups}\n\n"
        "See `reports/reconstruction_report.json` and "
        "`reports/quarantined_groups.jsonl` for details.\n"
    )
