from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterable
from typing import Any

from dataset_unchunker.loaders import INTERNAL_SOURCE_SPLIT
from dataset_unchunker.schema import QuarantineEntry, UnchunkConfig


def group_id_for(row: dict[str, Any], config: UnchunkConfig) -> str:
    if config.group_id_column:
        return str(row.get(config.group_id_column, ""))
    return "|".join(str(row.get(column, "")) for column in config.group_by)


def validate_required_columns(row: dict[str, Any], config: UnchunkConfig) -> None:
    missing = sorted(column for column in config.required_columns if column not in row)
    if missing:
        raise ValueError(f"missing required columns: {', '.join(missing)}")


def collect_groups(
    rows: Iterable[dict[str, Any]],
    config: UnchunkConfig,
) -> tuple[OrderedDict[str, list[dict[str, Any]]], int, list[QuarantineEntry]]:
    groups: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()
    quarantined: list[QuarantineEntry] = []
    input_rows = 0
    for row in rows:
        input_rows += 1
        try:
            validate_required_columns(row, config)
        except ValueError as exc:
            quarantined.append(
                QuarantineEntry(
                    group_id="<unknown>",
                    reason="missing_required_columns",
                    details={"error": str(exc)},
                    rows=1,
                )
            )
            continue
        group_id = group_id_for(row, config)
        if config.max_groups is not None and group_id not in groups and len(groups) >= config.max_groups:
            continue
        groups.setdefault(group_id, []).append(row)
    return groups, input_rows, quarantined


def source_split_values(rows: list[dict[str, Any]], source_split_column: str) -> list[str]:
    values: list[str] = []
    for row in rows:
        value = row.get(INTERNAL_SOURCE_SPLIT, row.get(source_split_column))
        if value is not None:
            values.append(str(value))
    return values
