from __future__ import annotations

import hashlib
from collections import Counter
from typing import Any, Protocol

from dataset_unchunker.grouping import source_split_values
from dataset_unchunker.loaders import INTERNAL_SOURCE_SPLIT
from dataset_unchunker.schema import (
    QuarantineEntry,
    ReconstructedGroup,
    SplitPolicy,
    Strategy,
    UnchunkConfig,
)


class TokenizerLike(Protocol):
    def encode(self, text: str, add_special_tokens: bool = False, **kwargs: Any) -> list[int]: ...

    def decode(
        self,
        token_ids: list[int],
        skip_special_tokens: bool = True,
        clean_up_tokenization_spaces: bool = False,
    ) -> str: ...

    def __call__(self, text: str, add_special_tokens: bool = True, **kwargs: Any) -> Any: ...


def load_tokenizer(name: str | None) -> TokenizerLike | None:
    if not name:
        return None
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise ImportError("token-overlap reconstruction requires transformers") from exc
    return AutoTokenizer.from_pretrained(name, use_fast=True)


def _as_int(value: Any, column: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{column} must be an integer") from exc


def _validate_and_sort_chunks(rows: list[dict[str, Any]], config: UnchunkConfig) -> list[dict[str, Any]]:
    indexed = [(_as_int(row.get(config.chunk_index_column), config.chunk_index_column), row) for row in rows]
    totals = {_as_int(row.get(config.chunk_total_column), config.chunk_total_column) for row in rows}
    if len(totals) != 1:
        raise ValueError("chunk_total is inconsistent within group")
    total = next(iter(totals))
    indexes = [index for index, _row in sorted(indexed, key=lambda item: item[0])]
    expected = list(range(total))
    if indexes != expected:
        raise ValueError(f"chunk indexes are incomplete or duplicated: expected {expected}, got {indexes}")
    return [row for _index, row in sorted(indexed, key=lambda item: item[0])]


def _find_overlap_sequence(left: list[int], right: list[int], max_overlap: int, expected: int | None) -> int | None:
    limit = min(max_overlap, len(left), len(right))
    if expected is not None:
        if expected > limit:
            return None
        return expected if left[-expected:] == right[:expected] or expected == 0 else None
    for size in range(limit, 0, -1):
        if left[-size:] == right[:size]:
            return size
    return None


def _find_overlap_text(left: str, right: str, max_overlap: int, expected: int | None) -> int | None:
    limit = min(max_overlap, len(left), len(right))
    if expected is not None:
        if expected > limit:
            return None
        return expected if left[-expected:] == right[:expected] or expected == 0 else None
    for size in range(limit, 0, -1):
        if left[-size:] == right[:size]:
            return size
    return None


def _token_count(tokenizer: TokenizerLike, text: str) -> int:
    try:
        encoded = tokenizer(text, add_special_tokens=True, verbose=False)
    except TypeError:
        encoded = tokenizer(text, add_special_tokens=True)
    return len(encoded.input_ids)


def _encode(tokenizer: TokenizerLike, text: str) -> list[int]:
    try:
        return tokenizer.encode(text, add_special_tokens=False, verbose=False)
    except TypeError:
        return tokenizer.encode(text, add_special_tokens=False)


def _decode(tokenizer: TokenizerLike, token_ids: list[int]) -> str:
    return tokenizer.decode(
        token_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )


def _reconstruct_token_overlap(
    texts: list[str],
    config: UnchunkConfig,
    tokenizer: TokenizerLike,
) -> tuple[str, int, list[int]]:
    merged = _encode(tokenizer, texts[0])
    overlaps: list[int] = []
    for text in texts[1:]:
        token_ids = _encode(tokenizer, text)
        overlap = _find_overlap_sequence(merged, token_ids, config.max_overlap, config.expected_overlap)
        if overlap is None:
            raise ValueError("token overlap could not be inferred")
        overlaps.append(overlap)
        merged.extend(token_ids[overlap:])
    reconstructed = _decode(tokenizer, merged)
    return reconstructed, _token_count(tokenizer, reconstructed), overlaps


def _reconstruct_text_overlap(texts: list[str], config: UnchunkConfig) -> tuple[str, list[int]]:
    merged = texts[0]
    overlaps: list[int] = []
    for text in texts[1:]:
        overlap = _find_overlap_text(merged, text, config.max_overlap, config.expected_overlap)
        if overlap is None:
            raise ValueError("text overlap could not be inferred")
        overlaps.append(overlap)
        merged += text[overlap:]
    return merged, overlaps


def _select_split(rows: list[dict[str, Any]], config: UnchunkConfig, group_id: str) -> str:
    values = source_split_values(rows, config.source_split_column)
    if config.split_policy == SplitPolicy.TRAIN or not values:
        return "train"
    if config.split_policy == SplitPolicy.ORIGINAL_IF_CONSISTENT:
        unique = sorted(set(values))
        return unique[0] if len(unique) == 1 else "train"
    if config.split_policy == SplitPolicy.MAJORITY:
        counts = Counter(values)
        return sorted(counts, key=lambda value: (-counts[value], value))[0]
    digest = hashlib.sha256(group_id.encode("utf-8")).digest()
    return "validation" if digest[0] < 26 else "train"


def _base_row(
    group_id: str,
    rows: list[dict[str, Any]],
    config: UnchunkConfig,
) -> tuple[dict[str, Any], dict[str, list[Any]]]:
    excluded = {
        config.text_column,
        INTERNAL_SOURCE_SPLIT,
        *config.chunk_metadata_columns,
    }
    first = rows[0]
    output: dict[str, Any] = {}
    conflicts: dict[str, list[Any]] = {}
    for key, first_value in first.items():
        if key in excluded:
            continue
        values = [row.get(key) for row in rows]
        if all(value == first_value for value in values):
            output[key] = first_value
        else:
            sample = []
            for value in values:
                if value not in sample:
                    sample.append(value)
                if len(sample) >= 5:
                    break
            conflicts[key] = sample
    output["reconstructed_group_id"] = group_id
    return output, conflicts


def reconstruct_group(
    group_id: str,
    rows: list[dict[str, Any]],
    config: UnchunkConfig,
    tokenizer: TokenizerLike | None = None,
) -> ReconstructedGroup | QuarantineEntry:
    try:
        sorted_rows = _validate_and_sort_chunks(rows, config)
        texts = [str(row.get(config.text_column) or "") for row in sorted_rows]
        strategy = config.resolved_strategy
        token_count: int | None = None
        overlaps: list[int] = []
        if strategy == Strategy.TOKEN_OVERLAP:
            if tokenizer is None:
                raise ValueError("token-overlap requires a tokenizer")
            text, token_count, overlaps = _reconstruct_token_overlap(texts, config, tokenizer)
        elif strategy == Strategy.TEXT_OVERLAP:
            text, overlaps = _reconstruct_text_overlap(texts, config)
            if tokenizer is not None:
                token_count = _token_count(tokenizer, text)
        elif strategy == Strategy.CONCAT_NO_OVERLAP:
            text = "".join(texts)
            if tokenizer is not None:
                token_count = _token_count(tokenizer, text)
        else:
            raise ValueError(f"unsupported strategy: {strategy}")
        base, conflicts = _base_row(group_id, sorted_rows, config)
        base[config.text_column] = text
        base["split"] = _select_split(sorted_rows, config, group_id)
        base["source_chunk_count"] = len(sorted_rows)
        if token_count is not None:
            base["reconstructed_token_count"] = token_count
        base["reconstruction_status"] = "reconstructed"
        return ReconstructedGroup(group_id=group_id, row=base, conflicts=conflicts, inferred_overlaps=overlaps)
    except ValueError as exc:
        return QuarantineEntry(
            group_id=group_id,
            reason="unsafe_reconstruction",
            details={"error": str(exc)},
            rows=len(rows),
        )
