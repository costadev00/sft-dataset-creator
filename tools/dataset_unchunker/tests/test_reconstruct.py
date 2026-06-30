from __future__ import annotations

from dataset_unchunker.reconstruct import reconstruct_group
from dataset_unchunker.schema import QuarantineEntry, Strategy, UnchunkConfig


class CharTokenizer:
    def encode(self, text, add_special_tokens=False, **kwargs):
        return [ord(char) for char in text]

    def decode(self, token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False):
        return "".join(chr(token_id) for token_id in token_ids)

    def __call__(self, text, add_special_tokens=True, **kwargs):
        class Encoded:
            input_ids = [ord(char) for char in text]

        return Encoded()


def config(**kwargs) -> UnchunkConfig:
    values = {
        "text_column": "text",
        "chunk_index_column": "chunk_index",
        "chunk_total_column": "chunk_total",
        "group_by": ("doc_id",),
    }
    values.update(kwargs)
    return UnchunkConfig(**values)


def test_single_chunk_passes_through_without_chunk_columns() -> None:
    result = reconstruct_group(
        "doc-1",
        [{"doc_id": "doc-1", "text": "full text", "chunk_index": 0, "chunk_total": 1, "token_count": 9}],
        config(strategy=Strategy.TEXT_OVERLAP),
    )

    assert not isinstance(result, QuarantineEntry)
    assert result.row["text"] == "full text"
    assert "chunk_index" not in result.row
    assert "chunk_total" not in result.row
    assert "token_count" not in result.row
    assert result.row["source_chunk_count"] == 1


def test_token_overlap_reconstructs_with_tokenizer() -> None:
    result = reconstruct_group(
        "doc-1",
        [
            {"doc_id": "doc-1", "text": "abcdef", "chunk_index": 0, "chunk_total": 2},
            {"doc_id": "doc-1", "text": "defghi", "chunk_index": 1, "chunk_total": 2},
        ],
        config(strategy=Strategy.TOKEN_OVERLAP, tokenizer="dummy", max_overlap=3),
        CharTokenizer(),
    )

    assert not isinstance(result, QuarantineEntry)
    assert result.row["text"] == "abcdefghi"
    assert result.row["reconstructed_token_count"] == 9
    assert result.inferred_overlaps == [3]


def test_text_overlap_reconstructs_without_tokenizer() -> None:
    result = reconstruct_group(
        "doc-1",
        [
            {"doc_id": "doc-1", "text": "alpha beta", "chunk_index": 0, "chunk_total": 2},
            {"doc_id": "doc-1", "text": "beta gamma", "chunk_index": 1, "chunk_total": 2},
        ],
        config(strategy=Strategy.TEXT_OVERLAP, max_overlap=4),
    )

    assert not isinstance(result, QuarantineEntry)
    assert result.row["text"] == "alpha beta gamma"
    assert result.inferred_overlaps == [4]


def test_missing_chunk_is_quarantined() -> None:
    result = reconstruct_group(
        "doc-1",
        [
            {"doc_id": "doc-1", "text": "part 0", "chunk_index": 0, "chunk_total": 3},
            {"doc_id": "doc-1", "text": "part 2", "chunk_index": 2, "chunk_total": 3},
        ],
        config(strategy=Strategy.TEXT_OVERLAP),
    )

    assert isinstance(result, QuarantineEntry)
    assert result.reason == "unsafe_reconstruction"
    assert "incomplete" in result.details["error"]


def test_inconsistent_chunk_total_is_quarantined() -> None:
    result = reconstruct_group(
        "doc-1",
        [
            {"doc_id": "doc-1", "text": "part 0", "chunk_index": 0, "chunk_total": 2},
            {"doc_id": "doc-1", "text": "part 1", "chunk_index": 1, "chunk_total": 3},
        ],
        config(strategy=Strategy.TEXT_OVERLAP),
    )

    assert isinstance(result, QuarantineEntry)
    assert "inconsistent" in result.details["error"]


def test_conflicting_metadata_is_reported_and_omitted() -> None:
    result = reconstruct_group(
        "doc-1",
        [
            {"doc_id": "doc-1", "title": "A", "text": "abcdef", "chunk_index": 0, "chunk_total": 2},
            {"doc_id": "doc-1", "title": "B", "text": "defghi", "chunk_index": 1, "chunk_total": 2},
        ],
        config(strategy=Strategy.TEXT_OVERLAP, max_overlap=3),
    )

    assert not isinstance(result, QuarantineEntry)
    assert "title" not in result.row
    assert result.conflicts["title"] == ["A", "B"]


def test_concat_no_overlap_requires_explicit_zero_overlap() -> None:
    cfg = config(strategy=Strategy.CONCAT_NO_OVERLAP, expected_overlap=0)
    cfg.validate()
    result = reconstruct_group(
        "doc-1",
        [
            {"doc_id": "doc-1", "text": "abc", "chunk_index": 0, "chunk_total": 2},
            {"doc_id": "doc-1", "text": "def", "chunk_index": 1, "chunk_total": 2},
        ],
        cfg,
    )

    assert not isinstance(result, QuarantineEntry)
    assert result.row["text"] == "abcdef"
