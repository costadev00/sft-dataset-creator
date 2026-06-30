from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from dataset_unchunker.schema import SourceSpec


INTERNAL_SOURCE_SPLIT = "_dataset_unchunker_source_split"


def _token(token_env: str) -> str | None:
    value = os.getenv(token_env)
    return value if value else None


def _download_hub_file(source: SourceSpec) -> Path:
    if not source.repo_id or not source.source_file:
        raise ValueError("repo_id and source_file are required to download a Hub file")
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise ImportError("reading Hub files requires huggingface_hub") from exc
    path = hf_hub_download(
        repo_id=source.repo_id,
        repo_type="dataset",
        filename=source.source_file,
        revision=source.revision,
        token=_token(source.token_env),
    )
    return Path(path)


def _iter_parquet(path: Path, *, source_split: str | None = None) -> Iterator[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ImportError("reading parquet requires pyarrow") from exc
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches():
        for row in batch.to_pylist():
            if source_split is not None:
                row[INTERNAL_SOURCE_SPLIT] = source_split
            yield row


def _iter_dataset_splits(source: SourceSpec) -> Iterator[dict[str, Any]]:
    if not source.repo_id:
        raise ValueError("repo_id is required when source_file is not a local parquet")
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError("reading Hugging Face datasets requires datasets") from exc
    for split in source.splits:
        dataset = load_dataset(
            source.repo_id,
            source.config_name,
            split=split,
            streaming=True,
            revision=source.revision,
            token=_token(source.token_env),
        )
        for row in dataset:
            value = dict(row)
            value[INTERNAL_SOURCE_SPLIT] = split
            yield value


def iter_rows(source: SourceSpec) -> Iterator[dict[str, Any]]:
    if source.source_file:
        local_path = Path(source.source_file).expanduser()
        if local_path.exists():
            yield from _iter_parquet(local_path)
            return
        yield from _iter_parquet(_download_hub_file(source))
        return
    yield from _iter_dataset_splits(source)
